%{
Friday April 17 2015, testing l1l2 block version
This script only works if you have CVX installed to find
the reference answers.
  Stephen Becker, stephen.becker@colorado.edu
%}


%% Simple test for just l1 norm:
[Y,L,S,params] = loadSyntheticProblem('verySmall_l1');
% And test:
opts = struct('tol',1e-8);
[LL,SS,errHist] = solver_RPCA_Lagrangian(Y,params.lambdaL,params.lambdaS,[],opts);
fprintf(2,'Error with Lagrangian version, L1 norm, is %.2e (L), %.2e (S)\n\n', ...
    norm( LL - L, 'fro')/norm(L,'fro'),  norm( SS - S, 'fro')/norm(S,'fro') );

%% Test l1l2 block norm (sum of l2 norms of rows)

% -- Load reference solutiion
[Y,L,S,params] = loadSyntheticProblem('verySmall_l1l2');

% -- Solve Lagrangian version
opts = struct('tol',1e-10,'printEvery',50,'L1L2','rows');
normLS = sqrt(norm(L,'fro')^2+norm(S,'fro')^2);
% opts.FISTA = true; opts.restart = Inf; opts.BB = false; opts.quasiNewton = false;
opts.errFcn = @(LL,SS) sqrt(norm(LL-L,'fro')^2+norm(SS-S,'fro')^2)/normLS;
[LL,SS,errHist] = solver_RPCA_Lagrangian(Y,params.lambdaL,params.lambdaS,[],opts);
fprintf(2,'Error with Lagrangian version, L1L2 via rows, is %.2e (L), %.2e (S)\n\n', ...
    norm( LL - L, 'fro')/norm(L,'fro'),  norm( SS - S, 'fro')/norm(S,'fro') );

% -- Solve simplest constrained versions
opts = struct('tol',1e-10,'printEvery',50,'L1L2','rows');
normLS = sqrt(norm(L,'fro')^2+norm(S,'fro')^2);
% opts.FISTA = true; opts.restart = Inf; opts.BB = false; opts.quasiNewton = false;
opts.errFcn = @(LL,SS) sqrt(norm(LL-L,'fro')^2+norm(SS-S,'fro')^2)/normLS;
% opts.sum    = true; % doesn't work with L1L2 = 'rows'
opts.max    = true;
[LL,SS,errHist] = solver_RPCA_constrained(Y,params.lambdaMax, params.tauMax,[], opts);
fprintf(2,'Error with simple constrained version, L1L2 via rows, is %.2e (L), %.2e (S)\n\n', ...
    norm( LL - L, 'fro')/norm(L,'fro'),  norm( SS - S, 'fro')/norm(S,'fro') );

% --- Solve fancy constrained versions
opts = struct('tol',1e-12,'printEvery',50,'L1L2','rows');
normLS = sqrt(norm(L,'fro')^2+norm(S,'fro')^2);
% opts.FISTA = true; opts.restart = Inf; opts.BB = false; opts.quasiNewton = false;
opts.errFcn = @(LL,SS) sqrt(norm(LL-L,'fro')^2+norm(SS-S,'fro')^2)/normLS;
% opts.sum    = true; % doesn't work with L1L2 = 'rows'
opts.max    = true;
% opts.tau0   = params.tauMax; % cheating
[LL,SS,errHist,tau] = solver_RPCA_SPGL1(Y,params.lambdaMax, params.epsilon,[], opts);
fprintf(2,'\n\nError with fancier constrained version, L1L2 via rows, is %.2e (L), %.2e (S)\n\n', ...
    norm( LL - L, 'fro')/norm(L,'fro'),  norm( SS - S, 'fro')/norm(S,'fro') );


%% And suppose we want l1l2 on the columns?
% We can use the same reference solution by just transposing
[Y,L,S,params] = loadSyntheticProblem('verySmall_l1l2');
Y = Y';
L = L';
S = S';
opts = struct('tol',1e-10,'printEvery',150,'L1L2','cols');
normLS = sqrt(norm(L,'fro')^2+norm(S,'fro')^2);
opts.errFcn = @(LL,SS) sqrt(norm(LL-L,'fro')^2+norm(SS-S,'fro')^2)/normLS;
[LL,SS,errHist] = solver_RPCA_Lagrangian(Y,params.lambdaL,params.lambdaS,[],opts);
fprintf(2,'Error with Lagrangian version, L1L2 via columns, is %.2e (L), %.2e (S)\n\n', ...
    norm( LL - L, 'fro')/norm(L,'fro'),  norm( SS - S, 'fro')/norm(S,'fro') );

% --- Now simple constrained version
opts = struct('tol',1e-10,'printEvery',150,'L1L2','cols');
normLS = sqrt(norm(L,'fro')^2+norm(S,'fro')^2);
% opts.FISTA = true; opts.restart = Inf; opts.BB = false; opts.quasiNewton = false;
opts.errFcn = @(LL,SS) sqrt(norm(LL-L,'fro')^2+norm(SS-S,'fro')^2)/normLS;
% opts.sum    = true; % doesn't work with L1L2 = 'rows'
opts.max    = true;
[LL,SS,errHist] = solver_RPCA_constrained(Y,params.lambdaMax, params.tauMax,[], opts);
fprintf(2,'Error with simple constrained version, L1L2 via columns, is %.2e (L), %.2e (S)\n\n', ...
    norm( LL - L, 'fro')/norm(L,'fro'),  norm( SS - S, 'fro')/norm(S,'fro') );

% --- Now fancier constrained version
opts = struct('tol',1e-12,'printEvery',200,'L1L2','cols');
normLS = sqrt(norm(L,'fro')^2+norm(S,'fro')^2);
opts.errFcn = @(LL,SS) sqrt(norm(LL-L,'fro')^2+norm(SS-S,'fro')^2)/normLS;
opts.max    = true;
[LL,SS,errHist,tau] = solver_RPCA_SPGL1(Y,params.lambdaMax, params.epsilon,[], opts);
fprintf(2,'\n\nError with fancier constrained version, L1L2 via columns, is %.2e (L), %.2e (S)\n\n', ...
    norm( LL - L, 'fro')/norm(L,'fro'),  norm( SS - S, 'fro')/norm(S,'fro') );


%% Ignore this:
% The code I used to generate the reference solutions:

% m = 10;
% n = 11;
% rng(343);
% Y = randn(m,n);

% -- First, test Lagrangian version, just l1
% lambdaL = .4;
% lambdaS = .1;
% cvx_begin
%   cvx_precision best
%   variables L(m,n) S(m,n)
%   minimize lambdaL*norm_nuc(L)+lambdaS*norm(S(:),1)+.5*sum_square(vec(L+S-Y))
% cvx_end
% S( abs(S)<1e-10 )=0;
% save('~/Repos/fastRPCA/utilities/referenceSolutions/verySmall_l1.mat',...
%     'Y','L','S','lambdaL','lambdaS','m','n' );

% -- Now, test Lagrangian version with l1l2
% lambdaL = .25;
% lambdaS = .24;
% cvx_begin
%   cvx_precision best
%   variables L(m,n) S(m,n)
%   minimize lambdaL*norm_nuc(L)+lambdaS*sum(norms(S'))+.5*sum_square(vec(L+S-Y))
% cvx_end
% S( abs(S)<1e-10 )=0
% save('~/Repos/fastRPCA/utilities/referenceSolutions/verySmall_l1l2.mat',...
%     'Y','L','S','lambdaL','lambdaS','m','n' );
