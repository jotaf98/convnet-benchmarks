function net = alexnet_owt_simplenn(opts)
%ALEXNET_OWT_SIMPLENN
%   AlexNet variant from the One Weird Trick paper
%   http://arxiv.org/abs/1404.5997

  if nargin == 0
    opts = {'CudnnWorkspaceLimit', 1024*1024*1204} ; % 1GB
  end

  net.layers = {} ;

  net = add_block(net, [11, 11, 3, 64], 'stride', 4, 'pad', 2, 'opts', {opts}) ;
  
  net.layers{end+1} = struct('type', 'pool', 'method', 'max', 'pool', [3 3], 'stride', 2) ;

  net = add_block(net, [5, 5, 64, 192], 'pad', 2, 'opts', {opts}) ;
  
  net.layers{end+1} = struct('type', 'pool', 'method', 'max', 'pool', [3 3], 'stride', 2) ;

  
  net = add_block(net, [3, 3, 192, 384], 'pad', 1, 'opts', {opts}) ;
  
  net = add_block(net, [3, 3, 384, 256], 'pad', 1, 'opts', {opts}) ;
  
  net = add_block(net, [3, 3, 256, 256], 'pad', 1, 'opts', {opts}) ;
  
  net.layers{end+1} = struct('type', 'pool', 'method', 'max', 'pool', [3 3], 'stride', 2) ;

  
  net = add_block(net, [6, 6, 256, 4096], 'opts', {opts}) ;
  
  net = add_block(net, [1, 1, 4096, 4096], 'opts', {opts}) ;
  
  net = add_block(net, [1, 1, 4096, 1000], 'opts', {opts}) ;

  net.layers(end) = [] ;  % delete last ReLU
end

function net = add_block(net, sz, varargin)

  net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.01*randn(sz, 'single'), zeros(sz(4), 1, 'single')}}, ...
    varargin{:}) ;

  net.layers{end+1} = struct('type', 'relu') ;

end


