%% Demo of RPCA on webcam
% We demonstrate this on a video clip taken from a surveillance
% camera in a subway station.  Not only is there a background
% and a foreground (i.e. people), but there is an escalator
% with periodic motion.  Conventional background subtraction
% algorithms would have difficulty with the escalator, but 
% this RPCA formulation easily captures it as part of the low-rank
%   structure.
% The clip is taken from the data at this website (after
% a bit of processing to convert it to grayscale):
% http://perception.i2r.a-star.edu.sg/bk_model/bk_index.html



if ~exist('cam','var')
    cam = webcam(2)
end

frames = 20;

% Resolution goes in first dimension
% img = zeros(120*160*3,frameRate);
% 
for idx = 1:frames
     snap = snapshot(cam);
     snap = imresize(snap,[960/8 1280/8]);
     img(:,idx) = snap(:);
end

X = double(img);

min_func_opts.progTol = 1e-10;
min_func_opts.optTol  = 1e-10;
min_func_opts.MaxIter = 2;
min_func_opts.MaxFunEvals  = 2000;
min_func_opts.useMex  = 0;
min_func_opts.Corr    = 1;

m = 120;
n = 160;
r = 3;

U0 = randn(size(X,1),1);
V0 = randn(size(X,2),1);


while 1 > 0

snap = snapshot(cam);
snap = imresize(snap,[960/8 1280/8]);

X(:,1) = [];
X      = [X double(snap(:))];


%% Run the algorithm
%{
We solve
    min_{L,S}  max( ||L||_* , lambda ||S||_1 )
    subject to
            || L+S-X ||_F <= epsilon
%}
nFrames     = size(X,2);
% lambda      = 2e-2;
% L0          = repmat( median(X,2), 1, nFrames );
% S0          = X - L0;
% epsilon     = 5e-3*norm(X,'fro'); % tolerance for fidelity to data
% opts        = struct('sum',false,'L0',L0,'S0',S0,'max',true,...
%     'tau0',3e5,'SPGL1_tol',1e-1,'tol',1e-3);
% [L,S] = solver_RPCA_SPGL1(X,lambda,epsilon,[],opts);


%% Run Split-SPCP

% Set options for minFunc
%opts_min_func.progTol = 1e-10;
%opts_min_func.optTol  = 1e-10;
%opts_min_func.MaxIter = 2000;
%opts_min_func.MaxFunEvals  = 2000;
%opts_min_func.useMex  = 0;
%opts_min_func.Corr    = 1;

[U0,V0,S] = solver_split_SPCP(U0,V0,X,[],min_func_opts);

L = U0*V0.';

%% show all together in movie format
% If you run this in your own computer, you can see the movie. On the webpage,
%   we have a youtube version of the video.
% The top row is using equality constraints, and the bottom row
% is using inequality constraints.
%  The first column of both rows is the same (i.e. it is the original image).
mat  = @(x) reshape( x, m, n, 3 );
figure(1); clf;
%h = axes('position',[0  0  1  1]);
colormap( 'Gray' );
%for k = 1:nFrames
image( [uint8(mat(X(:,end))), uint8(mat(L(:,end))), 20*uint8(mat(S(:,end))) ] );
% compare it to just using the median
%     imagesc( [mat(X(:,k)), mat(L0(:,k)),  mat(S0(:,k))] );
axis image
xlabel('From left to right: Full image, background, foreground.')
set(gca,'xtick',[],'ytick',[],'fontsize',24,'position',[0 0 1 1],'units','normalized')
drawnow;
%end


end