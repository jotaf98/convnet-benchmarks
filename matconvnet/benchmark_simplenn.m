
% set to MatConvNet root folder
root = '../../matconvnet-bitbucket' ;

% set to GPU number
gpu = 0 ;


steps = 10 ;
dryruns = 10 ;

sz = [224, 224, 3, 128] ;

opts = {'ConserveMemory', false} ;

run([root '/matlab/vl_setupnn.m']) ;


% set up
net = alexnet_owt_simplenn() ;
net = vl_simplenn_tidy(net) ;

x = randn(sz, 'single') ;
der = ones(1, 1, 1000, sz(4), 'single') ;
res = [] ;

if ~isempty(gpu)
  gpuDevice(gpu) ;
  net = vl_simplenn_move(net, 'gpu') ;
  x = gpuArray(x) ;
  der = gpuArray(der) ;
end


% dry runs
for i = 1:dryruns
  res = vl_simplenn(net, x, der, res, opts{:}) ;
  gather(res(end).x) ;
end
for i = 1:dryruns
  res = vl_simplenn(net, x, [], res, opts{:}) ;
  gather(res(end).x) ;
end


% time forward-mode
tic() ;
for i = 1:steps
  res = vl_simplenn(net, x, [], res, opts{:}) ;
end
gather(res(end).x) ;
fwd_time = toc() ;


% time train-mode
tic() ;
for i = 1:steps
  res = vl_simplenn(net, x, der, res, opts{:}) ;
end
gather(res(end).x) ;
total_time = toc() ;


% print results
fprintf('Forward speed: %0.2f ms\n', fwd_time * 1000 / steps) ;
fprintf('Backward speed: %0.2f ms\n', (total_time - fwd_time) * 1000 / steps) ;
fprintf('Total speed (train mode): %0.2f ms\n', total_time * 1000 / steps) ;


