%  Very simple and intuitive neural network implementation
%
%  Carl Lndahl, 2008
%  email: carl(dot)londahl(at)gmail(dot)com
%  Feel free to redistribute and/or to modify in any way
%  Modified by David Contreras, 2020 to run about 30 percent faster with some simplifications and vectorization
%  email: dvd.cnt@gmail.com

function m = neural()
  % DATA SETS; demo file
  [Attributes, Classifications] = mendez;
  n = 2.6;
  nbrOfNodes = 8;
  nbrOfEpochs = 800;

  % Initialize matrices with random weights 0-1
  W = rand(nbrOfNodes, size(Attributes,2));
  U = rand(size(Classifications,2),nbrOfNodes);
  figure; hold on; e = size(Attributes,1);
  
  for m = 1: nbrOfEpochs
      % Iterate through all examples
      for i=1:e
          % Input data from current example set
          I = Attributes(i,:).';
          D = Classifications(i,:).';

          % Propagate the signals through network
          H = f(W*I);
          O = f(U*H);

          % Output layer error
          delta_i = O.*(1-O).*(D-O);

          % Calculate error for each node in layer_(n-1)
          delta_j = H.*(1-H).*(U.'*delta_i);

          % Adjust weights in matrices sequentially
          U = U + n.*delta_i*(H.');
          W = W + n.*delta_j*(I.');
      end
      
      % Calculate RMS error
      D = Classifications.';
      I = Attributes.';
      RMS_Err=norm(D-f(U*f(W*I)),1);    
      y = RMS_Err/e;
      plot(m,log(y),'*');
  end
end

function x = f(x)
  x = 1./(1+exp(-x));
end
