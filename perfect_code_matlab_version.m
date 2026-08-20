tic;

project_root=fileparts(mfilename('fullpath'));
addpath(project_root,fullfile(project_root,'source'),...
    fullfile(project_root,'visualization'));

generate_parameters();
load('parameters.mat','param');

yr=365*24*60*60;
checkpointer=param.checkpointer;
output_interval=param.output_interval;
checkpoint_interval=param.checkpoint_interval;
Nt=param.Nt;

alpha=param.alpha;
sina=sind(alpha);
cosa=cosd(alpha);
xsize=param.xsize;
ysize=param.ysize;
% The grid module owns Nx/Ny now: with stretching they follow from the core
% cell counts plus the arms, not from xsize/element_size. With no stretching
% requested it returns exactly the old uniform grid, Nx and Ny included.
g=build_stretched_grid(param);
Nx=g.Nx;
Ny=g.Ny;
N=(Nx+1)*(Ny+1)*2;
dx=g.dx_core;      % core spacing, for reporting and surface-station x
dy=g.dy_core;
dz=dy*sina;

if mod(Nx,2)==0
    error('BP3 requires an odd Nx so the fault lies on the central column.');
end

if ~checkpointer
    x=g.x;
    y=g.y;
    xp=g.xp;
    yp=g.yp;
    Xuy=y*cosa+xp;
    Yuy=y*sina+xp*0;
    Xux=yp*cosa+x;
    Yux=yp*sina+x*0;
    Xtau=y*cosa+x;
    Ytau=y*sina+x*0;
    Xsigma=yp(2:Ny,1)*cosa+xp(1,2:Nx);
    Ysigma=yp(2:Ny,1)*sina+xp(1,2:Nx)*0;
    z=y*sina;
    xd=y;
    save('coord.mat','x','y','xp','yp','Xtau','Xux','Xuy','Ytau',...
        'Yux','Yuy','Xsigma','Ysigma','z','xd','g');
else
    load('coord.mat');
end
% Spacings between adjacent nodes of each staggered field, for the stress
% recovery below. Uniform grid: every entry equals dx or dy.
sx_uy=diff(xp(:))';        % between uy columns -> at x positions,  1 x Nx
sy_ux=diff(yp(:));         % between ux rows    -> at y positions,  Ny x 1
sx_ux=diff(x(:))';         % between ux columns -> 1 x (Nx-1)
sy_uy=diff(y(:));          % between uy rows    -> (Ny-1) x 1

% Exact operators for the parts of the recovery that were hand-merged: the two
% node derivatives that were MATLAB gradient() (a two-cell secant inside, a
% two-point one-sided difference at the ends), the two midpoint-to-node
% interpolations that were 50/50 movmean, and the cell-centre-to-fault-node map
% whose END NODES were a copy of the nearest cell centre -- so sigma at the free
% surface came from half a cell down. Built once, because this feeds the
% timestep loop. recovery_operators.m records which direction of averaging was
% already exact and is deliberately left alone.
R=recovery_operators(x,y,xp,yp);

[rho,lambda,G,eta,K0,a,b,L,mu0,V0]=...
    build_layer_rsf(Nx,Ny,x,z,param);
lambdaP=movmean(movmean(lambda,2,2,'omitnan','Endpoints','discard'),...
    2,1,'Endpoints','discard');
GP=movmean(movmean(G,2,2,'omitnan','Endpoints','discard'),...
    2,1,'Endpoints','discard');

[sigman0,tau0,Pl0,Pr0]=initial_stress(0,0,0,0,alpha,K0,0,z,param);
P=zeros(Ny,1);

ux=zeros(Ny+1,Nx);
uy=zeros(Ny,Nx+1);
vx=zeros(Ny+1,Nx);
vy=zeros(Ny,Nx+1);
tauqs=zeros(Ny,Nx);
sigmaqs=zeros(Ny-1,Nx-1);
t=0;
dt=param.dt0;

fault_ix=(Nx+1)/2;
Gfault=(G(:,fault_ix-1)+G(:,fault_ix+1))/2;
% Local down-dip spacing at each fault node, not the core value: with a
% stretched y the fault can extend into the coarsened arm.
mfault=grid_metrics(g);
ksi=build_ksi(Gfault,L,mfault.uy.hyc(:),a,b,sigman0);
% At a domain ending at Wf, use the bottom fault node as the imposed
% deep-creep driver; all shallower nodes remain rate-and-state.
creep_mask=xd>=param.Wf;
rsf_mask=~creep_mask;

if ~checkpointer
    [sigma,tau,U,V,theta]=initial_fault(L,V0,param.Vinit,mu0,eta,...
        Ny,a,b,tau0,sigman0,0,param);
    V(creep_mask)=param.VL;
    LH=build_LH(lambda,G,sina,cosa,Nx,Ny,N,g,param);
    save('initiation.mat','LH','ksi','-v7.3');
    fid=fopen('output.txt','w+');
else
    load(['data_',int2str(checkpointer),'.mat']);
    load('initiation.mat','LH','ksi');
    fid=fopen('output.txt','a+');
end

global Um Vm taum sigmam Pm thetam dtm tm tm2
nout=ceil(Nt/output_interval);
Um=zeros(Ny,nout);
Vm=zeros(Ny,nout);
taum=zeros(Ny,nout);
sigmam=zeros(Ny,nout);
Pm=zeros(Ny,nout);
thetam=zeros(Ny,nout);
dtm=zeros(1,nout);
tm=zeros(1,nout);
tm2=zeros(1,nout);

surface_x=[-32e3,-16e3,-8e3,dx/2,-dx/2,8e3,16e3,32e3];
surface_side=[-1,-1,-1,1,-1,1,1,1];
surface_disp1=zeros(numel(surface_x),nout);
surface_disp2=zeros(numel(surface_x),nout);
surface_vel1=zeros(numel(surface_x),nout);
surface_vel2=zeros(numel(surface_x),nout);

dLH=decomposition(LH);
options=optimset('TolFun',param.friction_tolerance,'TolX',0);
toc;

for it=1:Nt
    drive=tauqs(rsf_mask,fault_ix)+tau0(rsf_mask);
    % Bracket the internal slip rate, which follows sign(Vp) = -motion_sign,
    % not motion_sign itself. Keying it off Vp keeps the bracket on the same
    % side as the drive even if a run overrides Vp directly.
    %
    % The width is twice the previous step's fastest rate-and-state point, not
    % a fixed ceiling. That contains the new root unless max|V| doubles in a
    % single step, whose measured worst case is 1.18 over ~700k steps of the
    % 100/50/25 m sets -- so a failure below is now a warning that the
    % velocity jumped, not an arbitrary cap being hit. The old fixed 10 m/s
    % stopped 50m_dip60_rev and yc40_dip60_rev mid-rupture at 9.97 m/s, where
    % Uphoff's dip-60 reverse itself reaches 7.8 m/s at the trace.
    % abs() first: the magnitude must stay positive, the branch carries sign.
    vb=2*max(abs(V(rsf_mask)));
    if param.Vp>0
        lower=zeros(nnz(rsf_mask),1);
        upper=zeros(nnz(rsf_mask),1)+vb;
    else
        lower=zeros(nnz(rsf_mask),1)-vb;
        upper=zeros(nnz(rsf_mask),1);
    end

    [Vrsf,~,exitflag]=bisection(@(VV) ...
        sigma(rsf_mask).*a(rsf_mask).*asinh(VV/(2*V0)...
        .*exp((mu0(rsf_mask)+b(rsf_mask)...
        .*log(V0*theta(rsf_mask)./L(rsf_mask)))./a(rsf_mask)))...
        +eta*VV-drive,lower,upper,zeros(nnz(rsf_mask),1),options);
    if any(exitflag<0) || any(~isfinite(Vrsf))
        error(['BP3 friction solve failed at iteration %d: the root left the '...
            'bracket of %g m/s, i.e. max|V| more than doubled in one step.'],...
            it,vb);
    end
    V(rsf_mask)=Vrsf;
    V(creep_mask)=param.VL;

    active=rsf_mask;
    speed=max(abs(V(active)),realmin);
    dt_stability=min(ksi(active).*L(active)./speed);
    dt=min([param.dtmax,param.dt_growth*dt,dt_stability,param.tfinal-t]);
    if dt<=0
        break;
    end

    q=abs(V)*dt./L;
    expo=q>1e-6;
    theta=expo.*(L./abs(V).*(1-exp(-q))+theta.*exp(-q))+...
        (~expo).*(theta+dt.*(1-abs(V).*theta./L));
    tau=tauqs(:,fault_ix)+tau0-eta*V;
    U=U+dt*V;

    RH=build_RH(lambda,G,sina,cosa,[],0,Nx,Ny,N,dx,dy,y,V,z,dz,param);
    S=dLH\RH;
    vpx=reshape(S(1:2:end),Ny+1,Nx+1);
    vpy=reshape(S(2:2:end),Ny+1,Nx+1);
    vy=vpy(1:Ny,:);
    vx=vpx(:,1:Nx);
    uy=uy+vy*dt;
    ux=ux+vx*dt;

    % Stress recovery. Every difference is divided by the spacing it actually
    % spans; the node derivatives and the midpoint-to-node interpolations come
    % from R (built above) instead of gradient() and movmean, which were exact
    % only on a uniform mesh -- and, at the boundary rows, not even there.
    %
    % Still plain differences, and correct as they stand: diff(uy,1,2)./sx_uy
    % and diff(ux,1,1)./sy_ux span a single interval between half-nodes. They
    % carry a (hp-hm)/4 position offset relative to the node they are assigned
    % to (about 1.3 m on the benchmark fault), which needs a wider stencil to
    % remove -- see recovery_operators.m.
    duxdx=R.Pyp2y*(ux*R.Dx.');       % d(ux)/dx at (y,x)
    duydy=(R.Dy*uy)*R.Pxp2x.';       % d(uy)/dy at (y,x)
    tauqs=G/sina.*(diff(uy,1,2)./sx_uy+...
        (1-2*cosa*cosa)*diff(ux,1,1)./sy_ux+...
        cosa*(duxdx-duydy));
    tauqs(:,fault_ix)=(tauqs(:,fault_ix-1)+tauqs(:,fault_ix+1))/2;

    % The two movmeans here go NODES -> MIDPOINTS, and xp/yp are the numerical
    % midpoints by construction, so 50/50 is exact. Left alone on purpose.
    sigmaqs=(lambdaP+2*GP).*diff(ux(2:Ny,:),1,2)./sx_ux+...
        lambdaP.*diff(uy(:,2:Nx),1,1)./sy_uy-...
        2*GP*cosa.*movmean(movmean(diff(ux,1,1)./sy_ux,2,2,...
        'Endpoints','discard'),2,1,'Endpoints','discard');
    sigmal=R.Psig*sigmaqs(:,(Nx-1)/2);
    sigmar=R.Psig*sigmaqs(:,(Nx+1)/2);
    sigma=sigman0-(sigmal+sigmar)/2;

    t=t+dt;
    fprintf(fid,['it=%d, t=%f yr, dt=%e s, max|V|=%e m/s, ',...
        'min sigma=%e Pa\n'],checkpointer+it,t/yr,dt,max(abs(V)),min(sigma));

    if mod(it,output_interval)==0
        write_memory(it,output_interval,U,V,tau,sigma,P,theta,dt,t,0,...
            tauqs,sigmaqs,uy,vy,ux,vx);
        io=it/output_interval;
        un=mean(ux(1:2,:),1);
        vn=mean(vx(1:2,:),1);
        us=interp1(x,un,surface_x,'linear');
        vs=interp1(x,vn,surface_x,'linear');
        uts=interp1(xp,uy(1,:),surface_x,'linear');
        vts=interp1(xp,vy(1,:),surface_x,'linear');
        if param.load_side_boundaries
            rigid_rate=zeros(size(surface_side));
        else
            rigid_rate=-surface_side*param.Vp/2;
        end
        surface_disp1(:,io)=(us+uts*cosa+rigid_rate*t*cosa)';
        surface_disp2(:,io)=(uts*sina+rigid_rate*t*sina)';
        surface_vel1(:,io)=(vs+vts*cosa+rigid_rate*cosa)';
        surface_vel2(:,io)=(vts*sina+rigid_rate*sina)';

        if param.live_plot && (io==1 || mod(io,param.live_plot_interval)==0)
            plot_bp3_live(param,xd,tm(1:io),Vm(:,1:io),Um(:,1:io),...
                taum(:,1:io),sigmam(:,1:io),dtm(1:io));
        end
    end

    if mod(it,checkpoint_interval)==0
        checkpoint_id=checkpointer+it;
        save(['data_',int2str(checkpoint_id),'.mat'],'U','V','tau',...
            'sigma','theta','dt','t','tauqs','sigmaqs','uy','vy','ux','vx');
        save('dataall.mat','Um','Vm','taum','sigmam','Pm','thetam',...
            'dtm','tm','tm2','-v7.3');
        disp(it);
        toc;
    end

    if t>=param.tfinal
        break;
    end
end

nwritten=floor(it/output_interval);
Um=Um(:,1:nwritten);
Vm=Vm(:,1:nwritten);
taum=taum(:,1:nwritten);
sigmam=sigmam(:,1:nwritten);
Pm=Pm(:,1:nwritten);
thetam=thetam(:,1:nwritten);
dtm=dtm(1:nwritten);
tm=tm(1:nwritten);
tm2=tm2(1:nwritten);
surface_disp1=surface_disp1(:,1:nwritten);
surface_disp2=surface_disp2(:,1:nwritten);
surface_vel1=surface_vel1(:,1:nwritten);
surface_vel2=surface_vel2(:,1:nwritten);
save(['data_BP3_QD_',datestr(now,'yyyy-mm-dd-HH-MM-ss'),'.mat'],...
    'Um','Vm','taum','sigmam','thetam','dtm','tm','xd','param',...
    'surface_x','surface_disp1','surface_disp2','surface_vel1',...
    'surface_vel2','-v7.3');
fclose(fid);
if nwritten>0
    write_bp3_outputs(param,xd,tm,Um,Vm,taum,sigmam,thetam,...
        surface_x,surface_disp1,surface_disp2,surface_vel1,surface_vel2);
end

function LH=build_LH(lambda,G,sina,cosa,Nx,Ny,N,g,param)
%BUILD_LH Elastostatic stiffness on the sheared, optionally stretched grid.
%
%   Geometry enters only through grid_metrics. The shear Jacobian and the
%   stretch Jacobian multiply, so every cos(alpha)/sin(alpha) term below is
%   unchanged and only the spacings become position dependent: dx and dy are
%   reassigned per node at the top of each variable's block. On a uniform grid
%   they take the same value everywhere and this reduces exactly to the
%   original constant-spacing operator, which is the regression test.
%
%   ACCURACY ON A STRETCHED GRID -- READ THIS BEFORE TRUSTING A STRETCHED RUN.
%   The second-derivative stencils use the asymmetric FLUX form rather than the
%   symmetric -2/+1/+1 weights (increment 2b), and those are correct and
%   conservative on any spacing.
%
%   The centred FIRST and MIXED derivatives carried an extra error term. Over
%   unequal intervals,
%
%       (u+ - u-)/(hm+hp) = u' + (hp-hm)/2 * u'' + ...
%
%   For a smooth map hp-hm = dzeta^2*s'', so that form is still SECOND-ORDER
%   CONVERGENT -- but its leading error coefficient is proportional to s'', the
%   CURVATURE OF THE MESH GRADING. Measured consequences on the dip-60
%   y-coarsening series, with the seam inside Wf (y_core = 20 km):
%     - the error is pinned at the seam, because stretch_r = 2 makes s'' jump
%       there, and grows with the grading ratio;
%     - sigma on the rate-and-state fault rose by up to +5 MPa and nucleation
%       was delayed by 62 yr, or suppressed entirely at nys <= 27;
%     - stretch_r = 3 makes s'' continuous at the seam and removed only ~1/3
%       of it, confirming the map exponent is not the remedy;
%     - every one of these errors is EXACTLY zero when hm == hp, which is why
%       test_LH_equivalence passes at 3e-15 and cannot see any of it, and they
%       all carry a cosa factor, so the dip-90 symmetry regression is blind too.
%
%   INCREMENT 2d. The exact three-point non-uniform weights are exact for
%   quadratics, so the u'' term is identically absent. The mixed derivatives
%   become the full 3x3 tensor product instead of four corners. Measured: this
%   changed the seam bump by ~1 %. It is correct, but it was NOT the problem.
%
%   INCREMENT 2e -- THE BIG ONE. The cross-variable coupling is two operators,
%       u2 equation:  (lambda+G) ( d1 d2 - cos(alpha) d1^2 ) u1
%       u1 equation:  (lambda+G) ( d1 d2 - cos(alpha) d2^2 ) u2
%   (Shang, solver.tex). They used to be SUMMED into one coefficient per node,
%   which is free on a uniform grid but hides that each needs its own
%   spacing-dependent weights. The second-derivative half is evaluated at a
%   HALF NODE from four columns/rows, and its weights were hard-coded as
%   (1,-1,-1,1)/(2h^2) -- the uniform set. On the nys = 36 seam spacing the
%   correct weights are not even antisymmetric and the hard-coded set is 25 %
%   wrong on a pure quadratic. That is an order of magnitude larger than
%   anything increment 2d touched, and it is where the y-coarsening bump lives.
%   Now split, with source/fdweights.m supplying the exact weights. Verified to
%   reproduce the old merged coefficients to 3e-16 on a uniform grid.
%
%   All stencils now come from ONE helper, source/fdweights.m -- weights on
%   arbitrarily spaced nodes by polynomial exactness. onesided3 and centred3
%   are thin wrappers over it.
%
%   INCREMENT 2c: the last two row families that assumed LOCALLY uniform
%   spacing are now derived for arbitrary spacing.
%
%     - Fault traction continuity (ix==fault_ix normal, ix==fault_ix+1 shear)
%       differenced across the fault WITHOUT dividing by a spacing, which is
%       only valid when the cells either side are equal. Each side is now
%       divided by its own interval and the row rescaled by the centred
%       spacing, so the row keeps its previous magnitude.
%     - The free-surface rows used the uniform one-sided weights
%       (-3,4,-1)/(2dy). They now use the general three-point one-sided
%       weights for spacings (h1,h2), which collapse to those on a uniform
%       grid.
%
%   Both reduce EXACTLY to the previous expressions when h1==h2, so
%   test_LH_equivalence still pins them against the frozen reference.
    m=grid_metrics(g);
    % One-sided three-point d/dy weights at the free surface, for the two
    % staggered y grids. uy sits on g.y (first node ON the surface); ux sits on
    % g.yp (nodes straddling it).
    [wy1,wy2,wy3]=onesided3(m.uy.hyp(1),m.uy.hyp(2));
    hux1=m.ux.hyp(1);   % ux node spacing across the surface
    % The core-width guard is now belt-and-braces rather than load-bearing --
    % the stencils above are correct on a non-uniform mesh -- but a core
    % narrower than the stencils it contains is pathological, so keep failing
    % loudly.
    if g.nxs>0 && g.nxc<4
        error('build_LH:narrowCoreX', ...
            ['x core is %d cells per side; the fault stencils reach 2 cells, ' ...
             'so a stretched x grid needs at least 4.'],g.nxc);
    end
    if g.nys>0 && g.nyc<4
        error('build_LH:narrowCoreY', ...
            ['y core is %d cells; the one-sided free-surface stencil reaches ' ...
             '2 cells, so a stretched y grid needs at least 4.'],g.nyc);
    end
    % TRIPLETS PER ROW, not stencil width -- the stencil is 17 NODES either way
    % (9 own-variable + 8 cross-variable), as in the stencil figure.
    %   before 2d/2e : 5 second-derivative + 4 mixed corners + 8 coupling = 17
    %   after        : 5 + 9 (mixed 3x3) + 12 (coupling split) = 26
    % The extra triplets land on columns that are already in the stencil, and
    % sparse() sums duplicates, so nnz per row stays 17. 28 gives a little
    % slack over the counted 26 because the overflow check below is fatal.
    ntrip=28;
    I=zeros(ntrip*N,1);
    J=zeros(ntrip*N,1);
    LL=zeros(ntrip*N,1);
    ik=1;
    for ix=1:Nx+1
        for iy=1:Ny+1
            kux=((ix-1)*(Ny+1)+iy-1)*2+1;
            kuy=kux+1;
            if (iy<Ny+1)
                % uy lives at (xp, y). dx,dy are the centred spacings, which is
                % what first derivatives need; hxm/hxp/hym/hyp are the one-sided
                % neighbour distances, which second derivatives need.
                dx=m.uy.hxc(ix); dy=m.uy.hyc(iy);
                hxm=m.uy.hxm(ix); hxp=m.uy.hxp(ix);
                hym=m.uy.hym(iy); hyp=m.uy.hyp(iy);
                if (ix==1) % far-left boundary uy=0
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=1;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=1;ik=ik+1;
                elseif (ix==Nx+1) % far-right boundary uy=0
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=1;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy-(Ny+1)*2;LL(ik)=1;ik=ik+1;
                elseif (iy==1 && ix~=(Nx+1)/2 && ix~=(Nx+1)/2+1)
                    % free surface: sigma_zz=0
                    % Both fault columns are excluded here so each falls
                    % through to its own branch below -- the slip-rate jump at
                    % ix==(Nx+1)/2 and shear-traction continuity at
                    % ix==(Nx+1)/2+1 -- the same pairing the interior rows use.
                    % Giving the surface row two sigma_zz=0 rows instead left
                    % the trace unconstrained: its elastic slip then crept away
                    % from the friction-solved value without bound, reaching
                    % 50x by 48 yr at dx=100 m. Applying the jump to the minus
                    % side alone fixes that but breaks the x -> -x reflection
                    % symmetry; both sides together fix it and keep the dip-90
                    % antisymmetry exact. build_RH must loop iy=1:Ny.
                    % (dx/G)*sigma_yy=0 at x^2=0:
                    % d2(u2)+lambda/(lambda+2G)d1(u1)
                    % -2G*cos(alpha)/(lambda+2G)d1(u2)=0.
                    % The normal derivative is one-sided at the surface;
                    % the other terms are evaluated at x^2=0.
                    lambda_surface=mean(lambda(1,[ix-1,ix]),'omitnan');
                    G_surface=mean(G(1,[ix-1,ix]),'omitnan');
                    normal_scale=dx/G_surface;
                    nsc=normal_scale*(lambda_surface+2*G_surface);
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=nsc*wy1;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy+2;LL(ik)=nsc*wy2;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy+4;LL(ik)=nsc*wy3;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy-(Ny+1)*2;LL(ik)=normal_scale*G_surface*cosa/dx;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=-normal_scale*G_surface*cosa/dx;ik=ik+1;
                    I(ik)=kuy;J(ik)=kux;LL(ik)=normal_scale*lambda_surface/(2*dx);ik=ik+1;
                    I(ik)=kuy;J(ik)=kux+2;LL(ik)=normal_scale*lambda_surface/(2*dx);ik=ik+1;
                    I(ik)=kuy;J(ik)=kux-(Ny+1)*2;LL(ik)=-normal_scale*lambda_surface/(2*dx);ik=ik+1;
                    I(ik)=kuy;J(ik)=kux-(Ny+1)*2+2;LL(ik)=-normal_scale*lambda_surface/(2*dx);ik=ik+1;
                elseif (iy==Ny && ix==(Nx+1)/2) % creeping bottom fault jump
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=-1;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=1;ik=ik+1;
                elseif (iy==Ny) % far-bottom boundary uy=0 away from fault
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=1;ik=ik+1;
                elseif (ix==(Nx+1)/2) % fault diff(vy)=Vy
%                     I(ik)=kuy;J(ik)=kuy;LL(ik)=-2;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=2;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy-(Ny+1)*2;LL(ik)=1;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy+2*(Ny+1)*2;LL(ik)=-1;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=-1;ik=ik+1;
                    I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=1;ik=ik+1;
                elseif (ix==(Nx+1)/2+1) % fault continous tau?
                    GA=G(iy,ix-2);GB=G(iy,ix);GC=(GA+GB)/2;
                    % Each side's d(uy)/dx is a one-sided difference over ITS
                    % OWN interval -- the minus side spans columns ix-2..ix-1,
                    % so its spacing is hxm(ix-1), not hxm(ix). Rescaled by the
                    % centred spacing dx so the row keeps the magnitude it had
                    % when both sides were equal (where sm=sp=1).
                    sm=dx/m.uy.hxm(ix-1);
                    sp=dx/m.uy.hxp(ix);
                    I(ik)=kuy;J(ik)=kuy-2*(Ny+1)*2;LL(ik)=sm*GA/GC;ik=ik+1; % uy3
                    I(ik)=kuy;J(ik)=kuy-(Ny+1)*2;LL(ik)=-sm*GA/GC;ik=ik+1; % uy4
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=-sp*GB/GC;ik=ik+1; % uy5
                    I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=sp*GB/GC;ik=ik+1; % uy6
                    % The cos(alpha)*d(uy)/dy terms, one per side. cA and cB
                    % already carry the 1/(2dy), so the bracketed weights are
                    % just the derivative stencil: centred (-1,0,+1) in the
                    % interior, one-sided (-3,+4,-1) at the free surface where
                    % iy-1 does not exist. Duplicate (I,J) pairs are summed by
                    % sparse(), so overlapping with uy4/uy5 above is fine.
                    cA=cosa/2/dy*dx*GA/GC;    % minus side, column ix-1
                    cB=-cosa/2/dy*dx*GB/GC;   % plus side, column ix
                    % Without the 1/(2dy): the one-sided weights carry their own
                    % spacing, so cA0*wy1 == -3*cA when h1==h2.
                    cA0=cosa*dx*GA/GC;
                    cB0=-cosa*dx*GB/GC;
                    if (iy==1)
                        I(ik)=kuy;J(ik)=kuy-(Ny+1)*2;LL(ik)=cA0*wy1;ik=ik+1;
                        I(ik)=kuy;J(ik)=kuy-(Ny+1)*2+2;LL(ik)=cA0*wy2;ik=ik+1;
                        I(ik)=kuy;J(ik)=kuy-(Ny+1)*2+4;LL(ik)=cA0*wy3;ik=ik+1;
                        I(ik)=kuy;J(ik)=kuy;LL(ik)=cB0*wy1;ik=ik+1;
                        I(ik)=kuy;J(ik)=kuy+2;LL(ik)=cB0*wy2;ik=ik+1;
                        I(ik)=kuy;J(ik)=kuy+4;LL(ik)=cB0*wy3;ik=ik+1;
                    else
                        I(ik)=kuy;J(ik)=kuy-(Ny+1)*2-2;LL(ik)=-cA;ik=ik+1; % uy1
                        I(ik)=kuy;J(ik)=kuy-2;LL(ik)=-cB;ik=ik+1; % uy2
                        I(ik)=kuy;J(ik)=kuy-(Ny+1)*2+2;LL(ik)=cA;ik=ik+1; % uy7
                        I(ik)=kuy;J(ik)=kuy+2;LL(ik)=cB;ik=ik+1; % uy8
                    end
                    I(ik)=kuy;J(ik)=kux;LL(ik)=cosa/2*GB/GC;ik=ik+1; % ux3
                    I(ik)=kuy;J(ik)=kux+2;LL(ik)=cosa/2*GB/GC;ik=ik+1; % ux6
                    I(ik)=kuy;J(ik)=kux-(Ny+1)*2;LL(ik)=-cosa/2*(GA+GB)/GC+(1-2*cosa*cosa)/dy*dx*(GA-GB)/GC;ik=ik+1; % ux2
                    I(ik)=kuy;J(ik)=kux-(Ny+1)*2+2;LL(ik)=-cosa/2*(GA+GB)/GC+(1-2*cosa*cosa)/dy*dx*(GB-GA)/GC;ik=ik+1; % ux5
                    I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2;LL(ik)=cosa/2*GA/GC;ik=ik+1; % ux1
                    I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2+2;LL(ik)=cosa/2*GA/GC;ik=ik+1; % ux4
%                                     I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2;LL(ik)=(1-2*cosa*cosa)/dy*dx+cosa/2;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2+2;LL(ik)=-(1-2*cosa*cosa)/dy*dx+cosa/2;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux;LL(ik)=-(1-2*cosa*cosa)/dy*dx+cosa/2;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux+2;LL(ik)=(1-2*cosa*cosa)/dy*dx+cosa/2;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux-(Ny+1)*2;LL(ik)=-cosa;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux-(Ny+1)*2+2;LL(ik)=-cosa;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy-2;LL(ik)=cosa/2/dy*dx;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy+2;LL(ik)=-cosa/2/dy*dx;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy-(Ny+1)*2-2;LL(ik)=-cosa/2/dy*dx;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy-(Ny+1)*2+2;LL(ik)=cosa/2/dy*dx;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy-2*(Ny+1)*2;LL(ik)=1;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy-(Ny+1)*2;LL(ik)=-1;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2;LL(ik)=(1-2*cosa*cosa)/dy*dx;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2+2;LL(ik)=-(1-2*cosa*cosa)/dy*dx;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy;LL(ik)=-1;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=1;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux;LL(ik)=-(1-2*cosa*cosa)/dy*dx;ik=ik+1;
%                                     I(ik)=kuy;J(ik)=kux+2;LL(ik)=(1-2*cosa*cosa)/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kux+(Ny+1)*2;LL(ik)=cosa/4;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kux+(Ny+1)*2+2;LL(ik)=cosa/4;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kux-(Ny+1)*2;LL(ik)=-cosa/2;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kux-(Ny+1)*2+2;LL(ik)=-cosa/2;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kux-3*(Ny+1)*2;LL(ik)=cosa/4;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kux-3*(Ny+1)*2+2;LL(ik)=cosa/4;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy+(Ny+1)*2-2;LL(ik)=cosa/4/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy+(Ny+1)*2+2;LL(ik)=-cosa/4/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy-2;LL(ik)=cosa/4/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy+2;LL(ik)=-cosa/4/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy-(Ny+1)*2-2;LL(ik)=-cosa/4/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy-(Ny+1)*2+2;LL(ik)=cosa/4/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy-2*(Ny+1)*2-2;LL(ik)=-cosa/4/dy*dx;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy-2*(Ny+1)*2+2;LL(ik)=cosa/4/dy*dx;ik=ik+1;
%                 elseif (iy==1)
%                     I(ik)=kuy;J(ik)=kuy;LL(ik)=1;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy+2;LL(ik)=-1;ik=ik+1;
%                 elseif (iy==Ny)
%                     I(ik)=kuy;J(ik)=kuy;LL(ik)=1;ik=ik+1;
%                     I(ik)=kuy;J(ik)=kuy-2;LL(ik)=-1;ik=ik+1;
                else % uy-Navier
                    GA=G(iy,ix);GB=G(iy+1,ix);GC=(GA+GB)/2;
                    lambdaA=lambda(iy,ix);lambdaB=lambda(iy+1,ix);lambdaC=(lambdaA+lambdaB)/2;
                    % Second derivatives in asymmetric form (increment 2b). The
                    % row scale stays dx^2 = hxc^2, so the cross terms below are
                    % unchanged. Uniform grid: hxm=hxp=dx and hym=hyp=dy, giving
                    % back -2 / +1 / +1 and dx^2/dy^2 exactly.
                    I(ik)=kuy;J(ik)=kuy;LL(ik)=-dx*(1/hxm+1/hxp) ...
                        -dx*dx/(GC*dy)*((lambdaA+2*GA)/hym+(lambdaB+2*GB)/hyp);ik=ik+1; % uy5
                    I(ik)=kuy;J(ik)=kuy-(Ny+1)*2;LL(ik)=dx/hxm;ik=ik+1; % uy4
                    I(ik)=kuy;J(ik)=kuy+(Ny+1)*2;LL(ik)=dx/hxp;ik=ik+1; % uy6
                    I(ik)=kuy;J(ik)=kuy-2;LL(ik)=dx*dx*(lambdaA+2*GA)/(GC*dy*hym);ik=ik+1; % uy2
                    I(ik)=kuy;J(ik)=kuy+2;LL(ik)=dx*dx*(lambdaB+2*GB)/(GC*dy*hyp);ik=ik+1; % uy8
                    % INCREMENT 2d: mixed derivative as an EXACT non-uniform
                    % tensor product. The four-corner form is the plain centred
                    % difference in each direction, which carries an
                    % O(hp-hm)*f'' error whose coefficient is the CURVATURE of
                    % the mesh grading -- discontinuous at the seam, which is
                    % what pinned the normal-stress bump on the fault.
                    % centred3 removes that term. The stencil becomes 3x3; the
                    % five new weights are identically zero on a uniform grid,
                    % so the frozen-reference regression is untouched.
                    [axm,ax0,axp]=centred3(hxm,hxp);
                    [aym,ay0,ayp]=centred3(hym,hyp);
                    if m.uniform, ax0=0; ay0=0; end
                    sx=hxm+hxp;   % restores the previous +/-1 x-difference scale
                    Mv=[cosa*dx*(lambdaC+GC+2*GA)/GC, ...
                        cosa*dx*(lambdaC+GC+2*GC)/GC, ...
                        cosa*dx*(lambdaC+GC+2*GB)/GC];
                    wy=[aym ay0 ayp]; wx=[axm ax0 axp];
                    for ey=1:3
                        for ex=1:3
                            cxy=-Mv(ey)/2*wy(ey)*sx*wx(ex);
                            if cxy~=0
                                I(ik)=kuy;
                                J(ik)=kuy+(ex-2)*(Ny+1)*2+(ey-2)*2;
                                LL(ik)=cxy;ik=ik+1;   % uy1/3/7/9, +2/4/5/6/8 if graded
                            end
                        end
                    end
                    if (ix==2 || ix==Nx)
                        I(ik)=kuy;J(ik)=kux-(Ny+1)*2;LL(ik)=1/dy*dx*(lambdaA+GC)/GC;ik=ik+1; % ux2
                        I(ik)=kuy;J(ik)=kux-(Ny+1)*2+2;LL(ik)=-1/dy*dx*(lambdaB+GC)/GC;ik=ik+1; % ux6
                        I(ik)=kuy;J(ik)=kux;LL(ik)=-1/dy*dx*(lambdaA+GC)/GC;ik=ik+1; % ux3
                        I(ik)=kuy;J(ik)=kux+2;LL(ik)=1/dy*dx*(lambdaB+GC)/GC;ik=ik+1; % ux7
                    else
                        % INCREMENT 2e. The u1 coupling in the u2 equation is
                        %     (lambda+G) * ( d1 d2 - cos(alpha) d1^2 ) u1
                        % i.e. TWO operators. They used to be summed into one
                        % coefficient at ux2/ux3/ux6/ux7, which is free on a
                        % uniform grid but hides the fact that each needs its
                        % own spacing-dependent weights. Split here.
                        %
                        % (i) the mixed derivative d1 d2 u1: staggered, one
                        % interval each way, so it is already right on any mesh.
                        I(ik)=kuy;J(ik)=kux-(Ny+1)*2;LL(ik)=1/dy*dx*(lambdaA+GC)/GC;ik=ik+1; % ux2
                        I(ik)=kuy;J(ik)=kux-(Ny+1)*2+2;LL(ik)=-1/dy*dx*(lambdaB+GC)/GC;ik=ik+1; % ux6
                        I(ik)=kuy;J(ik)=kux;LL(ik)=-1/dy*dx*(lambdaA+GC)/GC;ik=ik+1; % ux3
                        I(ik)=kuy;J(ik)=kux+2;LL(ik)=1/dy*dx*(lambdaB+GC)/GC;ik=ik+1; % ux7
                        % (ii) -cos(alpha)*d1^2 u1: a SECOND derivative at the
                        % half node xp(ix), from the four ux columns
                        % x(ix-2..ix+1), averaged over the two y levels. The old
                        % code hard-coded the uniform weights (1,-1,-1,1)/(2dx^2)
                        % as +/-cosa*(lam+G)/GC/4. Those are 25 % wrong on the
                        % nys=36 seam spacing and are not even antisymmetric
                        % there. fdweights gives the exact set for any spacing
                        % and reproduces (1,-1,-1,1)/(2dx^2) when uniform.
                        wd2x=fdweights(g.xp(ix),[g.x(ix-2) g.x(ix-1) g.x(ix) g.x(ix+1)],2);
                        d2cA=-cosa*(lambdaA+GA)/GC*dx*dx/2;   % level iy
                        d2cB=-cosa*(lambdaB+GB)/GC*dx*dx/2;   % level iy+1
                        I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2;LL(ik)=d2cA*wd2x(1);ik=ik+1; % ux1
                        I(ik)=kuy;J(ik)=kux-2*(Ny+1)*2+2;LL(ik)=d2cB*wd2x(1);ik=ik+1; % ux5
                        I(ik)=kuy;J(ik)=kux-(Ny+1)*2;LL(ik)=d2cA*wd2x(2);ik=ik+1; % ux2
                        I(ik)=kuy;J(ik)=kux-(Ny+1)*2+2;LL(ik)=d2cB*wd2x(2);ik=ik+1; % ux6
                        I(ik)=kuy;J(ik)=kux;LL(ik)=d2cA*wd2x(3);ik=ik+1; % ux3
                        I(ik)=kuy;J(ik)=kux+2;LL(ik)=d2cB*wd2x(3);ik=ik+1; % ux7
                        I(ik)=kuy;J(ik)=kux+(Ny+1)*2;LL(ik)=d2cA*wd2x(4);ik=ik+1; % ux4
                        I(ik)=kuy;J(ik)=kux+(Ny+1)*2+2;LL(ik)=d2cB*wd2x(4);ik=ik+1; % ux8
                    end
                end
            else
                I(ik)=kuy;J(ik)=kuy;LL(ik)=1;ik=ik+1;
            end
            if (ix<Nx+1)
                % ux lives at (x, yp); same split as above.
                dx=m.ux.hxc(ix); dy=m.ux.hyc(iy);
                hxm=m.ux.hxm(ix); hxp=m.ux.hxp(ix);
                hym=m.ux.hym(iy); hyp=m.ux.hyp(iy);
                if (iy==1 && ix==(Nx+1)/2)
                    % Fault-line ghost at (x=0, y=-dy/2): on the fault plane,
                    % half a cell above the free surface. sigma_xz there IS the
                    % fault shear traction, so imposing sigma_xz=0 destroyed the
                    % trace node once a rupture reached the surface. Zero
                    % curvature along the fault instead -- constrains no
                    % derivative, so no traction at the singular corner. See the
                    % same branch in $HOME/BP3/source/build_LH.m.
                    %
                    % Non-uniform safe: a pure difference of three collinear
                    % values, no spacing enters, so it is identical on a
                    % stretched mesh. (The y core always starts at the surface,
                    % so these three nodes are uniformly spaced anyway.)
                    I(ik)=kux;J(ik)=kux;LL(ik)=1;ik=ik+1;
                    I(ik)=kux;J(ik)=kux+2;LL(ik)=-2;ik=ik+1;
                    I(ik)=kux;J(ik)=kux+4;LL(ik)=1;ik=ik+1;
                elseif (iy==1) % free surface: sigma_xz=0
                    % (dx/G)*sigma_xy=0 at x^2=0:
                    % d2(u1)+(1-2*cos(alpha)^2)d1(u2)
                    % +cos(alpha)[d2(u2)-d1(u1)]=0.
                    % d2(u1) and d1(u1) use the ux values straddling the
                    % surface; d2(u2) is one-sided at the surface.
                    G_surface=mean(G(1,max(1,min(Nx,[ix-1,ix]))),'omitnan');
                    shear_scale=dx/sina;
                    % d2(u1) straddles the surface, so its interval is the ux
                    % node spacing there, not the centred hyc. The cosa*d2(u2)
                    % term is one-sided on the uy grid and split over the two
                    % columns, hence cosa/2 times the one-sided weights --
                    % which is -3cosa/(4dy), cosa/dy, -cosa/(4dy) when uniform.
                    cw1=cosa/2*wy1; cw2=cosa/2*wy2; cw3=cosa/2*wy3;
                    I(ik)=kux;J(ik)=kux;LL(ik)=-shear_scale/hux1;ik=ik+1;
                    I(ik)=kux;J(ik)=kux+2;LL(ik)=shear_scale/hux1;ik=ik+1;
                    I(ik)=kux;J(ik)=kuy;LL(ik)=shear_scale*(cw1-(1-2*cosa*cosa)/dx);ik=ik+1;
                    I(ik)=kux;J(ik)=kuy+2;LL(ik)=shear_scale*cw2;ik=ik+1;
                    I(ik)=kux;J(ik)=kuy+4;LL(ik)=shear_scale*cw3;ik=ik+1;
                    I(ik)=kux;J(ik)=kuy+(Ny+1)*2;LL(ik)=shear_scale*(cw1+(1-2*cosa*cosa)/dx);ik=ik+1;
                    I(ik)=kux;J(ik)=kuy+(Ny+1)*2+2;LL(ik)=shear_scale*cw2;ik=ik+1;
                    I(ik)=kux;J(ik)=kuy+(Ny+1)*2+4;LL(ik)=shear_scale*cw3;ik=ik+1;
                    if ix==1
                        I(ik)=kux;J(ik)=kux;LL(ik)=shear_scale*cosa/2/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux+2;LL(ik)=shear_scale*cosa/2/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux+(Ny+1)*2;LL(ik)=-shear_scale*cosa/2/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux+(Ny+1)*2+2;LL(ik)=-shear_scale*cosa/2/dx;ik=ik+1;
                    elseif ix==Nx
                        I(ik)=kux;J(ik)=kux;LL(ik)=-shear_scale*cosa/2/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux+2;LL(ik)=-shear_scale*cosa/2/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux-(Ny+1)*2;LL(ik)=shear_scale*cosa/2/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux-(Ny+1)*2+2;LL(ik)=shear_scale*cosa/2/dx;ik=ik+1;
                    else
                        I(ik)=kux;J(ik)=kux+(Ny+1)*2;LL(ik)=-shear_scale*cosa/4/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux+(Ny+1)*2+2;LL(ik)=-shear_scale*cosa/4/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux-(Ny+1)*2;LL(ik)=shear_scale*cosa/4/dx;ik=ik+1;
                        I(ik)=kux;J(ik)=kux-(Ny+1)*2+2;LL(ik)=shear_scale*cosa/4/dx;ik=ik+1;
                    end
                elseif (iy==Ny+1) % far-bottom boundary ux=0
                    I(ik)=kux;J(ik)=kux;LL(ik)=1;ik=ik+1;
                    I(ik)=kux;J(ik)=kux-2;LL(ik)=1;ik=ik+1;
                elseif (ix==1) % left boundary ux=0
                    I(ik)=kux;J(ik)=kux;LL(ik)=1;ik=ik+1;
                elseif (ix==Nx) % right boundary ux=0
                    I(ik)=kux;J(ik)=kux;LL(ik)=1;ik=ik+1;
                elseif (ix==(Nx+1)/2) % fault continous sigma?
                    GA=G(iy,ix-1);GB=G(iy,ix+1);GC=(GA+GB)/2;
                    lambdaA=lambda(iy,ix-1);lambdaB=lambda(iy,ix+1);lambdaC=(lambdaA+lambdaB)/2;
%                     I(ik)=kux;J(ik)=kux+(Ny+1)*2;LL(ik)=-(lambda+2*G)/G;ik=ik+1;
%                     I(ik)=kux;J(ik)=kux-(Ny+1)*2;LL(ik)=(lambda+2*G)/G;ik=ik+1;
                    % Normal-traction continuity: each side's d(ux)/dx over its
                    % own interval, the row rescaled by dx=hxc so that on a
                    % uniform grid dx/hxm=dx/hxp=1 and these are the previous
                    % weights exactly.
                    I(ik)=kux;J(ik)=kux;LL(ik)=-dx*((lambdaA+2*GA)/hxm ...
                        +(lambdaB+2*GB)/hxp)/GC;ik=ik+1; % ux3
                    I(ik)=kux;J(ik)=kux+(Ny+1)*2;LL(ik)=dx*(lambdaB+2*GB)/(GC*hxp);ik=ik+1; % ux4
                    I(ik)=kux;J(ik)=kux-(Ny+1)*2;LL(ik)=dx*(lambdaA+2*GA)/(GC*hxm);ik=ik+1; % ux2
                    % 2-interval centred d/dy, so it gets the exact weights like
                    % every other. It vanishes when GA==GB, which is every
                    % homogeneous run including BP3 -- that is not a reason to
                    % leave it wrong, since a layered model would wake it up and
                    % the error would then be silent and mesh dependent.
                    % CA is set so CA*byp reproduces the previous +K exactly.
                    CA=2*(GA-GB)*cosa/GC*dx;
                    [bym,by0,byp]=centred3(hym,hyp);
                    if m.uniform, by0=0; end
                    I(ik)=kux;J(ik)=kux+2;LL(ik)=CA*byp;ik=ik+1; % ux5 added
                    I(ik)=kux;J(ik)=kux;  LL(ik)=CA*by0;ik=ik+1; % ux3 added (graded only)
                    I(ik)=kux;J(ik)=kux-2;LL(ik)=CA*bym;ik=ik+1; % ux1 added
%                     I(ik)=kux;J(ik)=kux+(Ny+1)*2-2;LL(ik)=2*cosa/dy*dx/4;ik=ik+1;
%                     I(ik)=kux;J(ik)=kux+(Ny+1)*2+2;LL(ik)=-2*cosa/dy*dx/4;ik=ik+1;
%                     I(ik)=kux;J(ik)=kux-(Ny+1)*2-2;LL(ik)=-2*cosa/dy*dx/4;ik=ik+1;
%                     I(ik)=kux;J(ik)=kux-(Ny+1)*2+2;LL(ik)=2*cosa/dy*dx/4;ik=ik+1;
                    I(ik)=kux;J(ik)=kuy;LL(ik)=-lambdaA/GC/dy*dx;ik=ik+1; % uy3
                    I(ik)=kux;J(ik)=kuy+(Ny+1)*2;LL(ik)=lambdaB/GC/dy*dx;ik=ik+1; % uy4
                    I(ik)=kux;J(ik)=kuy-2;LL(ik)=lambdaA/GC/dy*dx;ik=ik+1; % uy1
                    I(ik)=kux;J(ik)=kuy+(Ny+1)*2-2;LL(ik)=-lambdaB/GC/dy*dx;ik=ik+1; % uy2
                else % ux-Navier
                    GA=G(iy-1,ix);GB=G(iy,ix);GC=(GA+GB)/2;
                    lambdaA=lambda(iy-1,ix);lambdaB=lambda(iy,ix);lambdaC=(lambdaA+lambdaB)/2;
                    % Second derivatives in asymmetric form (increment 2b);
                    % reduces to -2 / +1 / +1 and dx^2/dy^2 on a uniform grid.
                    I(ik)=kux;J(ik)=kux;LL(ik)=-(lambdaC+2*GC)/GC*dx*(1/hxm+1/hxp) ...
                        -dx*dx/(GC*dy)*(GA/hym+GB/hyp);ik=ik+1; % ux5
                    I(ik)=kux;J(ik)=kux-(Ny+1)*2;LL(ik)=(lambdaC+2*GC)/GC*dx/hxm;ik=ik+1; % ux4
                    I(ik)=kux;J(ik)=kux+(Ny+1)*2;LL(ik)=(lambdaC+2*GC)/GC*dx/hxp;ik=ik+1; % ux6
                    I(ik)=kux;J(ik)=kux-2;LL(ik)=dx*dx*GA/(GC*dy*hym);ik=ik+1; % ux2
                    I(ik)=kux;J(ik)=kux+2;LL(ik)=dx*dx*GB/(GC*dy*hyp);ik=ik+1; % ux8
                    % INCREMENT 2d, ux block -- same exact tensor product as the
                    % uy mixed derivative above. See centred3.m.
                    [axm,ax0,axp]=centred3(hxm,hxp);
                    [aym,ay0,ayp]=centred3(hym,hyp);
                    if m.uniform, ax0=0; ay0=0; end
                    sx=hxm+hxp;
                    Mv=[cosa*dx*(lambdaA+GA+2*GC)/GC, ...
                        cosa*dx*(lambdaC+GC+2*GC)/GC, ...
                        cosa*dx*(lambdaB+GB+2*GC)/GC];
                    wy=[aym ay0 ayp]; wx=[axm ax0 axp];
                    for ey=1:3
                        for ex=1:3
                            cxy=-Mv(ey)/2*wy(ey)*sx*wx(ex);
                            if cxy~=0
                                I(ik)=kux;
                                J(ik)=kux+(ex-2)*(Ny+1)*2+(ey-2)*2;
                                LL(ik)=cxy;ik=ik+1;   % ux1/3/7/9, +2/4/5/6/8 if graded
                            end
                        end
                    end
                    if (iy==2 || iy==Ny)
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2;LL(ik)=1/dy*dx*(lambdaC+GB)/GC;ik=ik+1; % uy6
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2-2;LL(ik)=-1/dy*dx*(lambdaC+GA)/GC;ik=ik+1; % uy4
                        I(ik)=kux;J(ik)=kuy;LL(ik)=-1/dy*dx*(lambdaC+GB)/GC;ik=ik+1; % uy5
                        I(ik)=kux;J(ik)=kuy-2;LL(ik)=1/dy*dx*(lambdaC+GA)/GC;ik=ik+1; % uy3
                    else
                        % INCREMENT 2e, mirror of the uy block. The u2 coupling
                        % in the u1 equation is
                        %     (lambda+G) * ( d1 d2 - cos(alpha) d2^2 ) u2
                        % -- again two operators previously summed into one
                        % coefficient. Split, and the second derivative gets
                        % exact non-uniform weights.
                        %
                        % (i) mixed derivative d1 d2 u2: staggered, one interval
                        % each way, correct on any mesh.
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2;LL(ik)=1/dy*dx*(lambdaC+GB)/GC;ik=ik+1; % uy6
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2-2;LL(ik)=-1/dy*dx*(lambdaC+GA)/GC;ik=ik+1; % uy4
                        I(ik)=kux;J(ik)=kuy;LL(ik)=-1/dy*dx*(lambdaC+GB)/GC;ik=ik+1; % uy5
                        I(ik)=kux;J(ik)=kuy-2;LL(ik)=1/dy*dx*(lambdaC+GA)/GC;ik=ik+1; % uy3
                        % (ii) -cos(alpha)*d2^2 u2 at the half node yp(iy), from
                        % the four uy rows y(iy-2..iy+1), averaged over the two
                        % columns ix, ix+1. This is THE term the y-coarsening
                        % bump lives on: the hard-coded (1,-1,-1,1)/(2dy^2) is
                        % 25 % wrong at the nys=36 seam. Material follows the
                        % existing per-row A/B assignment.
                        wd2y=fdweights(g.yp(iy),[g.y(iy-2) g.y(iy-1) g.y(iy) g.y(iy+1)],2);
                        d2eA=-cosa*(lambdaA+GA)/GC*dx*dx/2;   % rows iy-2 and iy
                        d2eB=-cosa*(lambdaB+GB)/GC*dx*dx/2;   % rows iy-1 and iy+1
                        I(ik)=kux;J(ik)=kuy-2*2;LL(ik)=d2eA*wd2y(1);ik=ik+1; % uy1
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2-2*2;LL(ik)=d2eA*wd2y(1);ik=ik+1; % uy2
                        I(ik)=kux;J(ik)=kuy-2;LL(ik)=d2eB*wd2y(2);ik=ik+1; % uy3
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2-2;LL(ik)=d2eB*wd2y(2);ik=ik+1; % uy4
                        I(ik)=kux;J(ik)=kuy;LL(ik)=d2eA*wd2y(3);ik=ik+1; % uy5
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2;LL(ik)=d2eA*wd2y(3);ik=ik+1; % uy6
                        I(ik)=kux;J(ik)=kuy+2;LL(ik)=d2eB*wd2y(4);ik=ik+1; % uy7
                        I(ik)=kux;J(ik)=kuy+(Ny+1)*2+2;LL(ik)=d2eB*wd2y(4);ik=ik+1; % uy8
                    end
                end
            else
                I(ik)=kux;J(ik)=kux;LL(ik)=1;ik=ik+1;
            end
        end
    end
    % A silent overflow of the preallocation would corrupt the operator, so
    % fail loudly instead. Bump ntrip above if a new term trips this.
    if ik-1>numel(I)
        error('build_LH:tripletOverflow', ...
            ['emitted %d triplets but preallocated %d (%d per unknown); ' ...
             'raise ntrip.'],ik-1,numel(I),ntrip);
    end

    LH=sparse(I(1:ik-1),J(1:ik-1),LL(1:ik-1));
end

function RH=build_RH(lambda,G,sina,cosa,dPdt,Biot,Nx,Ny,N,dx,dy,y,V,z,dz,param)
% BP3-QD right-hand side: prescribed tangential velocity jump on the fault.
%
% GRID STRETCHING: nothing here depends on the mesh, and dx/dy are unused. Every
% row this function writes to is algebraic -- a prescribed velocity on a row
% whose LH coefficients are +/-1 (side and bottom boundaries, the fault jump) --
% so no spacing enters. That remains true only while those rows keep unit
% scaling; if a future change scales them by a local dx, the matching factor has
% to be applied here too.

RH=zeros(N,1);
fault_ix=(Nx+1)/2;

% Symmetric far-field plate loading. In the oblique basis, uy is the
% fault-parallel velocity component. The LH boundary row averages the ghost
% and interior values, so its right-hand side is twice the face velocity.
if isfield(param,'load_side_boundaries') && param.load_side_boundaries
    left_velocity=-param.Vp/2;
    right_velocity=param.Vp/2;
    for iy=1:Ny
        kuy_left=(iy-1)*2+2;
        kuy_right=(Nx*(Ny+1)+iy-1)*2+2;
        RH(kuy_left)=2*left_velocity;
        RH(kuy_right)=2*right_velocity;
    end
end

% For a compact total-velocity domain, continue the rigid velocities along
% the two halves of the bottom boundary. Otherwise the deep-creep jump is
% concentrated at the single fault/bottom node while its neighbors are
% pinned to zero, creating a large artificial stress concentration.
if isfield(param,'load_bottom_boundaries') && param.load_bottom_boundaries
    bottom_left=-param.VL/2;
    bottom_right=param.VL/2;
    for ix=2:Nx
        if ix==fault_ix
            continue;
        end
        kuy=((ix-1)*(Ny+1)+Ny-1)*2+2;
        if ix<fault_ix
            RH(kuy)=bottom_left;
        else
            RH(kuy)=bottom_right;
        end
    end
end

% iy starts at 1: the surface fault node carries the jump row too, matching the
% build_LH branch that excludes both fault columns from the sigma_zz=0 case.
% The bottom fault intersection (iy==Ny) is the deep-creep driver.
for iy=1:Ny
    kuy=((fault_ix-1)*(Ny+1)+iy-1)*2+2;
    RH(kuy)=V(iy);
end
end

function ksi=build_ksi(G,L,dy,a,b,sigman0)
%BUILD_KSI Lapusta-style adaptive-timestep coefficient, one per fault node.
%   dy may be a scalar or a per-node vector of the LOCAL down-dip spacing.
%   It must be the local value: on a y-stretched mesh part of the
%   rate-and-state fault can sit in the coarsened arm, and feeding the core
%   spacing everywhere understates the cell size there. That errs safe (too
%   small dy -> larger k1 -> smaller ksi -> smaller dt) but wastes steps and
%   is not the stiffness the arm actually has.
    k1=pi/4*G./dy.*L./a./sigman0;
    k2=(b-a)./a;
    k3=(k1-k2).^2/4-k1;
    k4=min(1./(k1-k2), 0.2);
    k5=min(1-k2./k1, 0.2);
    ksi=k4.*(k3>0)+k5.*(~(k3>0));
end

function [rho,lambda,G,eta,K0,a,b,L,mu0,V0]=build_layer_rsf(Nx,Ny,x,z,param)
% Homogeneous BP3-QD material and down-dip friction profile.

G0=param.rho*param.cs^2;
lambda0=2*G0*param.nu/(1-2*param.nu);

rho=zeros(Ny,Nx)+param.rho;
lambda=zeros(Ny,Nx)+lambda0;
G=zeros(Ny,Nx)+G0;

% The central column is the internal fault interface, not bulk material.
fault_ix=(Nx+1)/2;
rho(:,fault_ix)=nan;
lambda(:,fault_ix)=nan;
G(:,fault_ix)=nan;

eta=sqrt(param.rho*G0)/2;

xd=z/sind(param.alpha);
a=zeros(Ny,1)+param.amax;
a(xd<param.H)=param.a0;
transition=xd>=param.H & xd<param.H+param.h;
a(transition)=param.a0+(param.amax-param.a0)...
    .*(xd(transition)-param.H)/param.h;

b=zeros(Ny,1)+param.b0;
L=zeros(Ny,1)+param.L0;
mu0=zeros(Ny,1)+param.f0;
V0=param.V0;
K0=zeros(Ny,1);
end

function g=build_stretched_grid(param)
%BUILD_STRETCHED_GRID Uniform-core / power-law-stretched grid with metrics.
%
%   g = BUILD_STRETCHED_GRID(param) returns the node and half-node coordinates
%   for the sheared BP3 grid together with the mapping metric ds/dzeta at both,
%   so build_LH can be written on a UNIFORM computational grid and carry the
%   geometry in the metric rather than in variable spacings.
%
%   The map (after Pranger's garnet, experiments/SEAS/BP4_QD) is uniform over
%   the first n_core intervals and then a power-law arm of exponent r that
%   lands exactly on the outer boundary:
%
%       s(z) = b*z                             z <= n_core/n_tot
%       s(z) = b*z + (H_tot - b)*chi^r         otherwise
%       b    = n_tot*H_core/n_core
%       chi  = n_tot*(z - n_core/n_tot)/(n_tot - n_core)
%
%   s is continuous at the seam (s = H_core there) and exact at s(1) = H_tot.
%   The metric is analytic, and because chi^(r-1) vanishes at the seam for
%   r >= 2 the CELL SIZE is continuous across it too; only its slope jumps.
%
%   param fields (all optional, defaults reproduce the uniform grid exactly):
%     element_size      spacing inside the uniform core
%     xsize, ysize      total extents: x spans +/-xsize/2, y spans [0 ysize]
%     x_core            half-width of the uniform core in x   (default xsize/2)
%     y_core            extent of the uniform core in y       (default ysize)
%     nx_stretch        cells in EACH x arm                   (default 0)
%     ny_stretch        cells in the y arm                    (default 0)
%     stretch_r         exponent                              (default 2)
%
%   With nx_stretch = ny_stretch = 0 this returns bit-for-bit the grid
%   Main_code.m builds today, which is the first regression test.

es=param.element_size;
xsize=param.xsize;
ysize=param.ysize;
r=getfield_default(param,'stretch_r',2);
x_core=getfield_default(param,'x_core',xsize/2);
y_core=getfield_default(param,'y_core',ysize);
nxs=getfield_default(param,'nx_stretch',0);
nys=getfield_default(param,'ny_stretch',0);

% The y seam and the fault: history, not a live constraint.
%
% Coarsening ON the fault was once a wrong answer rather than a resolution
% trade-off, because build_LH's hand-merged centred first and mixed derivatives
% were only first-order once hm ~= hp, which pinned truncation error at the
% seam and let it accumulate with the load. Measured at dip 60, dx 100 m,
% Wf = 40 km, seam at y_core = 20 km, ON THE PRE-2e OPERATOR:
%
%   nys   max cell in Wf   sigma error on the fault   first event
%   250   100 m   (uniform)          --               133.2 yr
%   133   251 m                  +0.8 MPa             147.5 yr
%    90   410 m                  +1.5 MPa             159.8 yr
%    54   739 m                  +2.6 MPa             179.0 yr
%    36  1140 m                  +3.6 MPa             195.3 yr
%    27  1538 m                  +4.4 MPa             none in 200 yr
%
% Increment 2e split those coefficients and took the nys=36 nucleation error
% from 62 yr to 0.05 yr. The five-case benchmark now runs at y_core = 20 km and
% matches the references to within a few years, and on the fixed operator that
% configuration beats the release code's y_core = 40 km on every measure.
%
% The old error() guard, and the two warnings that replaced it, are therefore
% GONE. They quoted the pre-2e magnitudes above as if they were current, and
% they fired on every production run. The same applies to the "y_core = Wf
% exactly is not enough, leave a 5 km buffer" rule: that came from a +0.11 ->
% +1.76 MPa measurement on the unfixed operator.
%
% What remains true is the ordinary statement, and it needs no trap: cells
% below y_core are coarser than the core, so y_core sets deep-fault
% resolution. Choose it deliberately.

nxc=round(x_core/es);        % core cells per x arm
nyc=round(y_core/es);        % core cells in y
nxt=nxc+nxs;                 % cells per x arm, total
nyt=nyc+nys;

Nx=2*nxt+1;
Ny=nyt+1;
if mod(Nx,2)==0
    error('build_stretched_grid:evenNx','Nx must be odd; got %d.',Nx);
end

% Computational coordinates. x is symmetric on [-1,1], y one-sided on [0,1].
% x and y are THE nodes; xp and yp are their numerical midpoints.
%
% They used to be the map evaluated at computational half-nodes, which is the
% same thing on a uniform grid but NOT on a stretched one: for a quadratic map
%     s(z+h/2) - (s(z)+s(z+h))/2 = -(h^2/8)*s''
% so the half-node sits off the midpoint by an amount proportional to the
% curvature of the grading -- 0 in the uniform core, a constant -4.13 m through
% the arm at y_core = 20 km / nys = 36, stepping at the seam. Because the
% offset is a STEP, diff(yp) departs from the divisor hyc that build_LH uses by
% -3.8 % in the single cell at the seam, exactly where the normal-stress bump
% begins. Defining the half-nodes AS the midpoints makes diff(yp) == hyc
% identically, so the operator and the stress recovery in Main_code (which
% divides by diff(yp)) agree by construction instead of approximately.
%
% On a uniform grid this is a no-op TO ROUNDOFF, not bit-for-bit: the map is
% linear there so midpoint and half-node coincide analytically, but the two
% arithmetic routes differ by ~1 ulp (7e-12 m on a 45 km domain, 2e-16
% relative). That is far inside test_LH_equivalence's 1e-12 tolerance, and the
% m.uniform guard in build_LH still forces the graded centre weights to exactly
% zero so nnz cannot drift.
dzx=2/(Nx-1);
dzy=1/(Ny-1);
zx=linspace(-1,1,Nx);
zy=linspace(0,1,Ny)';

[x,bx]=map_sym(zx,x_core,xsize/2,nxc,nxs,r);
[y,by]=map_one(zy,y_core,ysize,nyc,nys,r);
xp=midpoints(x);
yp=midpoints(y);

g=struct();
g.x=x; g.xp=xp; g.y=y; g.yp=yp;
% bx,by are ds/dzeta at the NODES. The half-node metric is gone with the
% half-node coordinates: nothing consumed g.bxp/g.byp.
g.bx=bx; g.by=by;
g.dzx=dzx; g.dzy=dzy;                     % uniform computational spacings
g.Nx=Nx; g.Ny=Ny;
g.nxc=nxc; g.nyc=nyc; g.nxs=nxs; g.nys=nys; g.r=r;
g.stretched=(nxs>0)||(nys>0);
% Physical spacings, for reporting and for build_ksi (which needs the local
% on-fault value, not a single global dy).
g.dx=diff(x); g.dy=diff(y);
g.dy_core=es; g.dx_core=es;
end


function p=midpoints(v)
%MIDPOINTS Staggered nodes: the numerical midpoints of v, with one ghost
%   half-cell beyond each end. Orientation is preserved, so a row of x nodes
%   gives a row of xp and a column of y nodes gives a column of yp.
row=isrow(v);
v=v(:).';
p=[v(1)-0.5*(v(2)-v(1)), 0.5*(v(1:end-1)+v(2:end)), v(end)+0.5*(v(end)-v(end-1))];
if ~row
    p=p.';
end
end

function v=getfield_default(s,name,d)
if isfield(s,name) && ~isempty(s.(name))
    v=s.(name);
else
    v=d;
end
end


function [s,beta]=map_one(z,Hc,Ht,nc,ns,r)
%MAP_ONE One-sided map on z in [0,1], extendable a little outside for ghosts.
if ns==0 || Ht==Hc
    s=Ht*z;
    beta=Ht*ones(size(z));
    return;
end
nt=nc+ns;
b=nt*Hc/nc;
zs=nc/nt;
chi=nt*(z-zs)/(nt-nc);
inner=z<=zs;
s=zeros(size(z));
beta=zeros(size(z));
s(inner)=b*z(inner);
beta(inner)=b;
% Chain rule directly: d/dz of (Ht-b)*chi^r is (Ht-b)*r*chi^(r-1)*dchi/dz with
% dchi/dz = nt/(nt-nc). garnet writes this as b*(1+(Ht/alpha-1)*r*chi^(r-1))
% with alpha = Ht*Hc*(nt-nc)/(nc*(Ht-Hc)); the two are algebraically identical.
s(~inner)=b*z(~inner)+(Ht-b)*chi(~inner).^r;
beta(~inner)=b+(Ht-b)*r*chi(~inner).^(r-1)*nt/(nt-nc);
end


function [s,beta]=map_sym(z,Hc,Ht,nc,ns,r)
%MAP_SYM The same map folded about the origin, for z in [-1,1].
%   s(z) = sign(z)*f(|z|) so s is odd; beta = f'(|z|) is even, since the two
%   sign factors from the chain rule cancel.
[a,ba]=map_one(abs(z),Hc,Ht,nc,ns,r);
s=sign(z).*a;
beta=ba;
end

function [wm,w0,wp]=centred3(hm,hp)
%CENTRED3 Three-point centred first-derivative weights on unequal spacing.
%
%   [wm,w0,wp] = CENTRED3(hm,hp) gives
%
%       f'(s) ~ wm*f(s-hm) + w0*f(s) + wp*f(s+hp)
%
%   exact for quadratics and therefore second order for ANY hm, hp. With
%   hm == hp == h it returns (-1, 0, +1)/(2h), the plain centred difference,
%   so the centre weight vanishes and the operator is unchanged on a uniform
%   grid.
%
%   WHY THE CENTRE WEIGHT MATTERS. The plain (f+ - f-)/(hm+hp) form has
%
%       (f+ - f-)/(hm+hp) = f' + (hp-hm)/2 * f'' + O(h^2)
%
%   and for a smooth map hp-hm = dzeta^2*s'', so it is still second-order
%   CONVERGENT but its leading error is proportional to s'', the curvature of
%   the mesh grading. These weights are exact for quadratics, so that term is
%   identically absent.
%
%   Now a wrapper over FDWEIGHTS.

w=fdweights(0,[-hm 0 hp],1);
wm=w(1); w0=w(2); wp=w(3);
end

function w=fdweights(x0,p,m)
%FDWEIGHTS Finite-difference weights on arbitrarily spaced nodes.
%
%   w = FDWEIGHTS(x0,p,m) returns row vector w with
%
%       f^(m)(x0)  ~  sum_i w(i) * f(p(i))
%
%   exact for every polynomial up to degree numel(p)-1, for ANY node positions
%   p and ANY evaluation point x0. x0 need NOT be one of the nodes -- half-node
%   stencils are the reason this function exists.
%
%   Method: impose exactness on the monomials, i.e. solve A*w = e_m with
%   A(k+1,i) = (p(i)-x0)^k / k!. Bengt Fornberg (Math. Comp. 51 (1988) 699-706)
%   gives a stable O(n^2) recursion for the same weights; at the n = 3 and 4
%   used here the direct solve is the same answer in three lines. Nodes are
%   scaled to unit spread before the solve, which keeps the Vandermonde-like
%   system well conditioned when p spans kilometres.
%
%   This is the single source of truth for every derivative stencil on a
%   non-uniform mesh. ONESIDED3 and CENTRED3 are thin wrappers over it and
%   return exactly the textbook uniform weights when the spacings are equal:
%
%       fdweights(0,[0 h 2h],1)*2h   ->  (-3,  4, -1)      = onesided3
%       fdweights(0,[-h 0 h],1)*2h   ->  (-1,  0,  1)      = centred3
%       fdweights(0,[-1.5 -.5 .5 1.5]*h,2)*2h^2 -> (1,-1,-1,1)
%
%   The last one is the half-node second derivative that build_LH previously
%   hard-coded. On the nys = 36 seam spacing (100, 100, 116.5 m) the correct
%   weights are NOT antisymmetric -- the two middle weights differ by more than
%   a factor of two -- and the hard-coded set is 25 % wrong on a pure quadratic.

p=p(:).';
n=numel(p);
if m>=n
    error('fdweights:tooFewNodes', ...
        'need at least %d nodes for a derivative of order %d; got %d.', ...
        m+1,m,n);
end
d=p-x0;
s=max(abs(d));
if s==0
    error('fdweights:degenerate','all nodes coincide with the evaluation point.');
end
q=d/s;                       % scaled to unit spread: conditioning
A=zeros(n);
for k=0:n-1
    A(k+1,:)=q.^k/factorial(k);
end
b=zeros(n,1);
b(m+1)=1;
w=(A\b).'/s^m;               % undo the scaling: d/dx = (1/s) d/dq
end

function []=generate_parameters()
% Parameters for the SEAS BP3-QD benchmark.

yr=365*24*60*60;

param.checkpointer=0;
param.output_interval=10;
param.checkpoint_interval=1000;
param.Nt=200000;
param.live_plot=false;
param.live_plot_interval=5;

% Preliminary compact-domain run. Use 100 m or finer for production
% benchmark comparisons; the benchmark target spacing is 25 m.
param.element_size=200;
param.xsize=80e3;  % +/- 40 km from the fault
param.ysize=45e3;  % 40 km frictional fault + 5 km creeping buffer
param.Nx=round(param.xsize/param.element_size)+1;
param.Ny=round(param.ysize/param.element_size)+1;

% Geometry and sense of motion. Use +1 for thrust and -1 for normal.
param.alpha=60;
param.motion_sign=-1;

% Provenance written into the SEAS output headers. The benchmark spec lists
% modeler and date as required header fields; override in run_config.json if a
% particular submission needs different attribution.
param.code_name='indnuc staggered-grid QD';
param.code_version='';
param.modeler='Meng Li';

% Homogeneous elastic medium.
param.rho=2670;
param.nu=0.25;
param.cs=3.464e3;
param.Biot=0;

% BP3 rate-and-state friction.
param.sigma0=50e6;
param.a0=0.010;
param.amax=0.025;
param.b0=0.015;
param.L0=0.008;
param.V0=1e-6;
param.f0=0.6;
param.H=15e3;
param.h=3e3;
param.Wf=40e3;

% Vp, VL and Vinit follow motion_sign and are resolved after the overlay below,
% not here -- see the note at the bottom of this file.
param.load_side_boundaries=true;
param.velocity_bracket=10;

% Time integration.
param.tfinal=1500*yr;
param.dt0=1;
param.dtmax=0.1*yr;
param.dt_growth=1.2;
param.friction_tolerance=5;

% A run folder can supply case-specific values without editing this file.
% Grid counts are recomputed after applying the overrides.
config_file=fullfile(pwd,'run_config.json');
supplied={};
if isfile(config_file)
    config=jsondecode(fileread(config_file));
    names=fieldnames(config);
    supplied=names;
    for k=1:numel(names)
        if ~strcmp(names{k},'tfinal_years')
            param.(names{k})=config.(names{k});
        end
    end
    if isfield(config,'tfinal_years')
        param.tfinal=config.tfinal_years*yr;
    end
    param.Nx=round(param.xsize/param.element_size)+1;
    param.Ny=round(param.ysize/param.element_size)+1;
end

% Driving velocities, resolved here so a run_config.json that sets motion_sign
% actually reaches them. They used to be assigned above the overlay, so a config
% asking for thrust got thrust's friction bracket and header with the default
% normal loading -- the bracket and the drive then disagree and the friction
% solve fails outright.
%
% motion_sign is the benchmark's sense of motion (+1 thrust, -1 normal), and is
% also the sign the spec assigns to slip, slip rate and shear traction. The
% solver's internal slip variable is the jump uy(+) - uy(-) set by the fault row
% in build_LH, which is the NEGATIVE of the spec's delta = uy(-) - uy(+)
% (eq 6). So the internal drive carries -motion_sign and write_bp3_outputs
% negates slip and shear stress on the way back out. Both halves are needed:
% fixing only the output sign simulates the wrong sense, fixing only the drive
% labels it wrongly. At alpha=90 the two senses are mirror images so the error
% is invisible, but for a dipping fault they are different physical problems --
% Omega+ is the hanging wall, and normal faulting needs it to move down-dip,
% uy(+) > uy(-).
internal_sign=-param.motion_sign;
for f={'Vp','VL','Vinit'}
    if ~any(strcmp(supplied,f{1}))
        param.(f{1})=internal_sign*1e-9;
    end
end

save('parameters.mat','param');
end

function m=grid_metrics(g)
%GRID_METRICS Local spacings the stencils need, one entry per node.
%
%   The operator is assembled on a UNIFORM computational grid; all geometry
%   enters here. Two maps compose to make the physical grid:
%
%     shear      X = x + y*cos(psi),  Z = y*sin(psi)
%     stretch    x = X(zeta),         y = Y(eta)
%
%   and their Jacobians multiply,
%
%     d(X,Z)/d(zeta,eta) = [[1 cos(psi)];[0 sin(psi)]] * diag(bx,by),
%
%   so the shear algebra already hard-coded in build_LH is untouched and only
%   d/dx -> (1/bx) d/dzeta, d/dy -> (1/by) d/deta change. Because the stretch
%   is separable, the spacings depend on ix or iy alone, never on both -- that
%   is what keeps this a vector per direction rather than a field per node.
%
%   Returned, for each of the two staggered variables:
%     hxm,hxp   distance to the x-neighbour below/above  (indexed by ix)
%     hym,hyp   distance to the y-neighbour below/above  (indexed by iy)
%     hxc,hyc   the centred spacing (hm+hp)/2, for flux-form second derivatives
%
%   uy lives at (xp, y); ux lives at (x, yp) -- hence separate sets.
%
%   A GENERAL curvilinear mesh would break separability and need the full
%   metric tensor with cross terms; the intent is that only this function
%   changes when that day comes, and build_LH keeps consuming m.<var>.<h>.

m=struct();
m.uy=spacings(g.xp,g.y,g.Nx+1,g.Ny);
m.ux=spacings(g.x,g.yp,g.Nx,g.Ny+1);
m.uniform=~g.stretched;
end


function s=spacings(xv,yv,nx,ny)
s.hxm=pad_diff(xv(:),nx);
s.hxp=[s.hxm(2:end);s.hxm(end)];
s.hym=pad_diff(yv(:),ny);
s.hyp=[s.hym(2:end);s.hym(end)];
s.hxc=(s.hxm+s.hxp)/2;
s.hyc=(s.hym+s.hyp)/2;
end


function h=pad_diff(v,n)
%PAD_DIFF Backward spacing at each of n nodes; the first is copied from the
%   second so boundary rows never see a NaN. Boundary rows are algebraic
%   (Dirichlet or one-sided) and do not use a two-sided spacing anyway.
d=diff(v(:));
if numel(d)<n-1
    d(end+1:n-1)=d(end);
end
h=[d(1);d(1:n-1)];
end

function [sigma,tau,U,V,theta]=initial_fault(L,V0,Vi,mu0,eta,Ny,a,b,tau0,sigman0,theta0,param)
% BP3-QD initial slip, slip rate, and state (benchmark equations 24-26).

U=zeros(Ny,1);
V=zeros(Ny,1)+param.Vinit;

friction_at_initial=(tau0-eta*V)./sigman0;
argument=2*V0./V.*sinh(friction_at_initial./a);
if any(argument<=0)
    error('BP3 initial-state inversion produced a non-positive logarithm argument.');
end
theta=L./V0.*exp((a.*log(argument)-mu0)./b);

sigma=sigman0;
tau=tau0-eta*V;
end

function [sigman0,tau0,Pl0,Pr0]=initial_stress(rho,rhof,rhog,g,alpha,K0,Biot,z,param)
% BP3-QD uniform effective normal stress and steady initial shear stress.

Ny=numel(z);
sigman0=zeros(Ny,1)+param.sigma0;

steady_friction=param.amax*asinh(param.Vinit/(2*param.V0)...
    *exp((param.f0+param.b0*log(param.V0/abs(param.Vinit)))... 
    /param.amax));
tau0=zeros(Ny,1)+param.sigma0*steady_friction+...
    sqrt(param.rho*(param.rho*param.cs^2))/2*param.Vinit;

Pl0=zeros(Ny,1);
Pr0=zeros(Ny,1);
end

function [w1,w2,w3]=onesided3(h1,h2)
%ONESIDED3 Three-point one-sided first-derivative weights at the first node.
%
%   [w1,w2,w3] = ONESIDED3(h1,h2) gives
%
%       f'(s1) ~ w1*f(s1) + w2*f(s1+h1) + w3*f(s1+h1+h2)
%
%   second order for ANY h1, h2. With h1 == h2 == h it returns
%   (-3, 4, -1)/(2h), the uniform weights the free-surface rows in build_LH
%   carried before the mesh could be stretched -- which is why the uniform
%   regression against build_LH_reference still holds.
%
%   Now a wrapper over FDWEIGHTS, which is the single source of truth for
%   every non-uniform stencil in the operator. Kept as a named function
%   because the free-surface rows read better with it.

w=fdweights(0,[0 h1 h1+h2],1);
w1=w(1); w2=w(2); w3=w(3);
end

function R=recovery_operators(x,y,xp,yp)
%RECOVERY_OPERATORS Exact staggered-grid operators for the stress recovery.
%
%   R = RECOVERY_OPERATORS(x,y,xp,yp) returns sparse matrices that replace the
%   hand-merged `gradient` calls and 50/50 `movmean` averages in Main_code's
%   stress recovery. Each one is exact where the thing it replaces was only
%   first order (or, at the free surface, zeroth order) on a graded mesh.
%
%   Applied as, with ux (Ny+1 x Nx) at (yp,x) and uy (Ny x Nx+1) at (y,xp):
%
%       d(ux)/dx at (y ,x)  =  R.Pyp2y*(ux*R.Dx.')
%       d(uy)/dy at (y ,x)  =  (R.Dy*uy)*R.Pxp2x.'
%       sigma    at  y      =  R.Psig*sigmaqs(:,column)
%
%   WHY EACH ONE EXISTS
%
%   Dy, Dx -- first derivative at the NODES.  MATLAB's gradient(f,coord) uses
%   the two-cell secant (f(i+1)-f(i-1))/(coord(i+1)-coord(i-1)) (see
%   toolbox/matlab/datafun/gradient.m), which drops the f(i) term entirely and
%   is only first order once hm ~= hp; its error is ((hp-hm)/2)*f''. That is
%   the same defect increment 2e removed from build_LH, left behind here. The
%   exact three-point weights are what FDWEIGHTS returns, and they are
%   identical to the distance-weighted average of the two adjacent staggered
%   slopes -- so this is consistent with the staggered operator, not a
%   competing discretisation.
%
%   At the two end nodes gradient uses a TWO-point one-sided difference, first
%   order on any mesh including a uniform one. These use the three-point
%   one-sided stencil instead, so the change is NOT a no-op on a uniform grid:
%   it is second order at the boundary where the old code was first order.
%
%   Pyp2y, Pxp2x -- staggered midpoints back onto the nodes. Direction matters
%   and only one direction was wrong:
%
%     nodes -> midpoints   xp = midpoints(x) BY CONSTRUCTION, so a 50/50
%                          average is EXACT. Main_code's lambdaP/GP and the
%                          cos(alpha) term of sigmaqs go this way and are
%                          left alone.
%     midpoints -> nodes   y(j) is NOT the midpoint of yp(j), yp(j+1) on a
%                          graded mesh -- it sits (hp-hm)/4 away. A 50/50
%                          average therefore samples the field at the wrong
%                          place, first order in the grading.
%
%   The end rows come out at exactly 1/2 on their own, because midpoints()
%   builds the ghost half-cells symmetrically about the end nodes.
%
%   Psig -- cell-centred sigma onto the fault nodes. sigmaqs row k lives at
%   yp(k+1), so the interior nodes need the same distance weighting as above.
%   The ends needed more: the old code COPIED the nearest cell centre, so
%   sigma at the free-surface fault node was taken from half a cell down --
%   50 m at 100 m resolution, a zeroth-order error and by far the largest of
%   the three, present on a uniform mesh too. Here both ends are one-sided
%   quadratic extrapolations from the three nearest centres.
%
%   NOT CHANGED, deliberately: the single-interval staggered differences
%   diff(uy,1,2)./sx_uy and diff(ux,1,1)./sy_ux. These land at the midpoint of
%   two half-nodes, which is (hp-hm)/4 from the node they are assigned to, so
%   they carry the same kind of first-order position offset -- about 1.3 m on
%   the benchmark fault, worth ~0.1 %. Removing it needs a wider stencil than
%   two points, which changes the recovery's footprint; that is a separate
%   decision, not a bug fix.

x=x(:).'; xp=xp(:).'; y=y(:); yp=yp(:);
Nx=numel(x); Ny=numel(y);
if numel(xp)~=Nx+1 || numel(yp)~=Ny+1
    error('recovery_operators:size', ...
        'expected numel(xp)=Nx+1 and numel(yp)=Ny+1; got %d,%d for Nx=%d,Ny=%d.', ...
        numel(xp),numel(yp),Nx,Ny);
end

R=struct();
R.Dy=node_derivative(y);
R.Dx=node_derivative(x);
R.Pyp2y=mid_to_node(y);
R.Pxp2x=mid_to_node(x);
R.Psig=centres_to_nodes(y,yp);
end


function D=node_derivative(v)
%NODE_DERIVATIVE First derivative at each node from its own three-point
%   stencil: centred inside, one-sided at the two ends.
v=v(:); n=numel(v);
if n<3
    error('recovery_operators:tooFewNodes','need at least 3 nodes; got %d.',n);
end
ii=zeros(3*n,1); jj=ii; vv=ii; c=0;
for k=1:n
    if k==1
        idx=[1 2 3];
    elseif k==n
        idx=[n-2 n-1 n];
    else
        idx=[k-1 k k+1];
    end
    w=fdweights(v(k),v(idx),1);
    ii(c+1:c+3)=k; jj(c+1:c+3)=idx; vv(c+1:c+3)=w; c=c+3;
end
D=sparse(ii,jj,vv,n,n);
end


function P=mid_to_node(v)
%MID_TO_NODE Linear interpolation from the n+1 midpoints of v onto its n
%   nodes, weighted by the distance to each neighbour rather than 50/50.
v=v(:); n=numel(v);
h=diff(v);
hh=[h(1);h;h(end)];                 % n+1 spacings, ghost half-cells mirrored
w=hh(2:n+1)./(hh(1:n)+hh(2:n+1));   % weight on midpoint j; 1/2 at both ends
P=sparse([(1:n).';(1:n).'],[(1:n).';(2:n+1).'],[w;1-w],n,n+1);
end


function P=centres_to_nodes(v,vp)
%CENTRES_TO_NODES sigmaqs rows -- at vp(2:n), i.e. the n-1 cell centres --
%   onto the n nodes v. Distance weighted inside, one-sided quadratic
%   extrapolation at the two ends, where no centre brackets the node.
v=v(:); vp=vp(:); n=numel(v); m=n-1;
if m<3
    error('recovery_operators:tooFewCentres', ...
        'need at least 3 cell centres for the end extrapolations; got %d.',m);
end
h=diff(v);
ii=[]; jj=[]; vv=[];
we=fdweights(v(1),vp(2:4),0);       % free surface, from the three nearest
ii=[ii;1;1;1]; jj=[jj;1;2;3]; vv=[vv;we(:)];
for k=2:n-1
    w=h(k)/(h(k-1)+h(k));           % weight on the centre ABOVE the node
    ii=[ii;k;k]; jj=[jj;k-1;k]; vv=[vv;w;1-w];
end
% Columns 1..m of sigmaqs sit at vp(2..n), so the three centres nearest the
% deep end are vp(n-2:n) -- NOT vp(n+1), which is the ghost half-node past the
% last row and carries no sigma at all.
wl=fdweights(v(n),vp(n-2:n),0);     % deep end, from the three nearest
ii=[ii;n;n;n]; jj=[jj;m-2;m-1;m]; vv=[vv;wl(:)];
P=sparse(ii,jj,vv,n,m);
end

function write_bp3_outputs(param,xd,t,U,V,tau,sigma,theta,...
    surface_x,disp1,disp2,vel1,vel2)
% Write the ASCII products requested for the SEAS BP3-QD benchmark.
%
% Filenames follow the spec: the on-fault (fltst_dp*) and off-fault (srfst_fn*)
% time series carry NO extension, only the three slip/stress evolution profiles
% are named *.dat.  Headers carry the fields the spec does not mark optional --
% problem, code, modeler, date, element size, station location -- plus the
% per-column descriptions the server uses to interpret the file.

% Runs saved before the provenance fields existed still have to be writable.
if ~isfield(param,'code_name')
    param.code_name='indnuc staggered-grid QD';
end
if ~isfield(param,'modeler')
    param.modeler='Meng Li';
end

outdir='output_BP3_QD';
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fault_stations=[0,2.5,5,7.5,10,12.5,15,17.5,20,25,30,35]*1e3;
fault_names={'000','025','050','075','100','125','150','175','200',...
    '250','300','350'};
motion_name='normal';
if param.motion_sign>0
    motion_name='thrust';
end
nt=numel(t);

for k=1:numel(fault_stations)
    [~,iy]=min(abs(xd-fault_stations(k)));
    filename=fullfile(outdir,['fltst_dp',fault_names{k}]);
    fid=fopen(filename,'w');
    write_common_header(fid,param,motion_name,nt);
    fprintf(fid,'# location=on fault, %.1f km down-dip distance\n',...
        fault_stations(k)/1e3);
    fprintf(fid,'# Column #1 = Time (s)\n');
    fprintf(fid,'# Column #2 = Slip (m)\n');
    fprintf(fid,'# Column #3 = Slip rate (log10 m/s)\n');
    fprintf(fid,'# Column #4 = Shear stress (MPa)\n');
    fprintf(fid,'# Column #5 = Normal stress (MPa)\n');
    fprintf(fid,'# Column #6 = State (log10 s)\n');
    fprintf(fid,'# The line below lists the names of the data fields\n');
    fprintf(fid,'t slip slip_rate shear_stress normal_stress state\n');
    fprintf(fid,'# Here is the time-series data.\n');
    % Slip and shear stress are negated into the spec's convention: the solver
    % carries the jump uy(+)-uy(-), the spec's delta is uy(-)-uy(+) (eq 6).
    % Slip rate and state are log10 magnitudes and normal stress is unsigned by
    % this, so only these two columns flip.
    values=[t(:),-U(iy,:)',log10(max(abs(V(iy,:)'),realmin)),...
        -tau(iy,:)'/1e6,sigma(iy,:)'/1e6,...
        log10(max(theta(iy,:)',realmin))];
    fprintf(fid,'%21.13E %14.6E %14.6E %14.6E %14.6E %14.6E\n',...
        values');
    fclose(fid);
end

surface_names={'srfst_fn-32','srfst_fn-16','srfst_fn-08','srfst_fn+00',...
    'srfst_fn-00','srfst_fn+08','srfst_fn+16','srfst_fn+32'};
% Nominal station positions from the spec; fn+00 and fn-00 are the two sides of
% the fault trace, actually sampled at the adjacent staggered nodes x=+/-dx/2.
surface_nominal=[-32e3,-16e3,-8e3,0,0,8e3,16e3,32e3];
for k=1:numel(surface_names)
    filename=fullfile(outdir,surface_names{k});
    fid=fopen(filename,'w');
    write_common_header(fid,param,motion_name,nt);
    fprintf(fid,'# location=on surface, %+g km distance off-fault\n',...
        surface_nominal(k)/1e3);
    fprintf(fid,'# sampled at x=%+.4f km\n',surface_x(k)/1e3);
    fprintf(fid,'# Column #1 = Time (s)\n');
    fprintf(fid,'# Column #2 = Displacement 1 (m)\n');
    fprintf(fid,'# Column #3 = Displacement 2 (m)\n');
    fprintf(fid,'# Column #4 = Velocity 1 (m/s)\n');
    fprintf(fid,'# Column #5 = Velocity 2 (m/s)\n');
    fprintf(fid,'# The line below lists the names of the data fields\n');
    fprintf(fid,'t disp_1 disp_2 vel_1 vel_2\n');
    fprintf(fid,'# Here is the time-series data.\n');
    values=[t(:),disp1(k,:)',disp2(k,:)',vel1(k,:)',vel2(k,:)'];
    fprintf(fid,'%21.13E %14.6E %14.6E %14.6E %14.6E\n',values');
    fclose(fid);
end

profile_mask=xd<=param.Wf;
profile_indices=find(profile_mask);
stride=max(1,round(500/param.element_size));
profile_indices=profile_indices(1:stride:end);
if profile_indices(end)~=find(profile_mask,1,'last')
    profile_indices(end+1)=find(profile_mask,1,'last');
end

% Negative scales on slip and shear stress for the same eq (6) convention flip
% applied to the on-fault time series above; normal stress is unaffected.
write_profile(fullfile(outdir,'slip.dat'),param,'slip','Slip (m)',...
    xd(profile_indices),t,V,U(profile_indices,:),-1);
write_profile(fullfile(outdir,'shear_stress.dat'),param,'shear_stress',...
    'Shear stress (MPa)',xd(profile_indices),t,V,...
    tau(profile_indices,:),-1e-6);
write_profile(fullfile(outdir,'normal_stress.dat'),param,'normal_stress',...
    'Normal stress (MPa)',xd(profile_indices),t,V,...
    sigma(profile_indices,:),1e-6);
end

function write_common_header(fid,param,motion_name,nt)
fprintf(fid,'# This is the file header:\n');
fprintf(fid,'# problem=SEAS Benchmark BP3-QD\n');
fprintf(fid,'# code=%s\n',param.code_name);
if isfield(param,'code_version') && ~isempty(param.code_version)
    fprintf(fid,'# version=%s\n',param.code_version);
end
fprintf(fid,'# modeler=%s\n',param.modeler);
fprintf(fid,'# date=%s\n',datestr(now,'yyyy/mm/dd'));
fprintf(fid,'# element size=%g m\n',param.element_size);
fprintf(fid,'# motion=%s\n',motion_name);
fprintf(fid,'# dip angle=%g degrees\n',param.alpha);
fprintf(fid,'# num time steps=%d\n',nt);
end

function write_profile(filename,param,field_name,description,xd,t,V,field,scale)
n=numel(xd);
fid=fopen(filename,'w');
fprintf(fid,'# This is the file header:\n');
fprintf(fid,'# problem=SEAS Benchmark BP3-QD\n');
fprintf(fid,'# modeler=%s\n',param.modeler);
fprintf(fid,'# date=%s\n',datestr(now,'yyyy/mm/dd'));
fprintf(fid,'# code=%s\n',param.code_name);
if isfield(param,'code_version') && ~isempty(param.code_version)
    fprintf(fid,'# code version=%s\n',param.code_version);
end
fprintf(fid,'# element size=%g m\n',param.element_size);
fprintf(fid,'# Row #1 = Distance down dip (m) with two zeros first\n');
fprintf(fid,'# Column #1 = Time (s)\n');
fprintf(fid,'# Column #2 = Max slip rate (log10 m/s)\n');
fprintf(fid,'# Columns #3-%d = %s\n',n+2,description);
fprintf(fid,['# Computational domain size: down-dip %g km, ' ...
    'distance off fault %g km, dip %g degrees\n'], ...
    param.ysize/1e3,param.xsize/2e3,param.alpha);
fprintf(fid,'# The line below lists the names of the data fields\n');
fprintf(fid,'xd\n');
fprintf(fid,'t max_slip_rate %s\n',field_name);
fprintf(fid,'# Here are the data\n');
fprintf(fid,'%14.6E ',[0,0,xd(:)']);
fprintf(fid,'\n');
for it=1:numel(t)
    fprintf(fid,'%21.13E %14.6E ',t(it),log10(max(abs(V(:,it)))));
    fprintf(fid,'%14.6E ',field(:,it)*scale);
    fprintf(fid,'\n');
end
fclose(fid);
end

function write_memory(it,output_interval,U,V,tau,sigma,P,theta,dt,t,t2,tauqs,sigmaqs,uy,vy,ux,vx)
global Um Vm taum sigmam Pm thetam dtm tm tm2 taumall sigmamall uymall vymall uxmall vxmall
Um(:,it/output_interval)=U;
Vm(:,it/output_interval)=V;
taum(:,it/output_interval)=tau;
sigmam(:,it/output_interval)=sigma;
Pm(:,it/output_interval)=P;
thetam(:,it/output_interval)=theta;
dtm(it/output_interval)=dt;
tm(it/output_interval)=t;
tm2(it/output_interval)=t2;
% taumall(:,:,it/output_interval)=tauqs;
% sigmamall(:,:,it/output_interval)=sigmaqs;
% uymall(:,:,it/output_interval)=uy;
% vymall(:,:,it/output_interval)=vy;
% uxmall(:,:,it/output_interval)=ux;
% vxmall(:,:,it/output_interval)=vx;
end