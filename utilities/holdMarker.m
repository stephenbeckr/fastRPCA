function symb = holdMarker(reset,symbList2)
%HOLDMARKER  Returns a new symbol for use in plotting or rotates through
% symb = holdMarker()
%   returns a new symbol for use in plotting
%
% holdMarker('reset') 
%   will reset the rotation
%   (if there is no output, then the rotation is reset)
% holdMarker( {j} )
%   will reset the rotation to start at symbol position "j"
%
%   Use this with plot( ..., 'marker', holdMarker() )
%
% holdMarker('reset',symbList) sets the symbol list (should be cell array)
%
% holdMarker( h )
%   will apply this function to the handles in h
%   (if h=0, uses h = get(gca,'children') )
%   ex.
%       h = plot( myMatrix );
%       holdMarker();  % reset
%       holdMarker( h ); % apply to all the lines in the plot -- each
%                        % will have a unique symbol.
%
% see also:
% set(gcf,'DefaultAxesColorOrder',[1 0 0;0 1 0;0 0 1],...
%   'DefaultAxesLineStyleOrder','-|--|:')
%   (using a 0 instead of gcf will make it permanent for your matlab session)
% Stephen Becker, May 2011
%
%   See also add_few_markers.m

persistent currentLocation symbListP

if isempty( currentLocation)
    currentLocation = 1;
end

if isempty( symbListP )
    %symbList = { '+','o','*','.','x','s',...
    %'d','^','v','>','<','p','h' };
    
    % Exclude the "."
    symbList = { '+','o','*','x','s',...
        'd','^','v','>','<','p','h' };
else
%     symbList = symbListP;
    % May 2013
    if ~iscell( symbListP )
        symbList = cell(size(symbListP));
        for jj = 1:length( symbListP )
            symbList{jj} = symbListP(jj);
        end
    else
        symbList = symbListP;
    end
end

% April 2013
if nargin > 0 && isequal(reset,0)
    reset = get( gca, 'children' );
end

do_reset = false;
if nargin > 0
    if ischar(reset)
        do_reset = true;
    elseif iscell( reset )
        currentLocation = reset{1};
    elseif all( ishandle(reset) ) && ~( isscalar(reset) && ( reset == 1) )
        % we have handles
        handles     = reset;
        for h = handles(:)'
            symb    = get_symbol;
            set( h, 'Marker', symb );
        end
    elseif isscalar(reset)
        currentLocation = reset;
    else
        error('invalid input');
    end
end
% if nargout == 0 || do_reset
if false;
    do_reset  = true;
else
    symb = get_symbol;
end

if do_reset
    currentLocation = 1;
    if nargin >= 2
        symbListP = symbList2;
    else
        symbListP = [];
    end
    return;
end



% This subfunction has side-effects
function symb = get_symbol
    if currentLocation > length(symbList), currentLocation = 1; end
    symb    = symbList{ currentLocation };
    currentLocation = currentLocation + 1;
end



end % end of main function
