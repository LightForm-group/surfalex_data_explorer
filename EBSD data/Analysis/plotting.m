% EBSD figures for Surfalex material using MTEX

clear

% Specify Crystal Symmetries

CS = {... 
  'notIndexed',...
  crystalSymmetry('m-3m', [4.05 4.05 4.05], 'mineral', 'Aluminium', 'color', 'light blue')};
setMTEXpref('defaultColorMap',parula);

%% Specify File Names

file_names = ["RD-ND", "RD-TD", "TD-ND"];
rotations = [[0, 90, 0], [0, 0, 90], [0, 90, 0]];

plot_ipf = true;
plot_pf = true;
plot_odf = true;
show_figures = false;

%% Set figure preferences

setMTEXpref('FontSize', 30);
setMTEXpref('FontName', 'SansSerif')
setMTEXpref('figSize', 'large');
setMTEXpref('outerPlotSpacing', 40);
setMTEXpref('innerPlotSpacing', 20)

for file_number = 1:length(file_names)
    % create an EBSD variable containing the data
    fname = sprintf('EBSD Data/Data/%1$s plane/%1$s Data.ctf', file_names(file_number));
    ebsd = EBSD.load(fname, CS, 'interface', 'ctf', 'convertEuler2SpatialReferenceFrame');

    % Rotate the map. Rotation angle depends on which plane is being considered
    phi = rotations(file_number * 3 - 2);
    theta = rotations(file_number * 3 - 1 );
    psi = rotations(file_number * 3);
    ebsd = rotate(ebsd, rotation('euler', phi * degree, theta * degree, psi * degree), 'keepXY');

    %% Plot the Band contrast and IPFZ maps together.
    if plot_ipf == true
        % Calculate grains from map
        ebsd = ebsd('indexed');
        [grains, ebsd.grainId] = calcGrains(ebsd, 'angle', 5 * degree);
        grains = smooth(grains);

        % Plotting convention for ipf map
        setMTEXpref('xAxisDirection','east');
        setMTEXpref('zAxisDirection','intoplane');

        fig = figure();
        plot(ebsd, ebsd.bc);
        colormap gray
        hold on
        oM = ipfHSVKey(ebsd('indexed'));
        oM.inversePoleFigureDirection = zvector;
        color = oM.orientation2color(ebsd('indexed').orientations);
        plot(ebsd('indexed'),color);
        hold off
        file_name = sprintf('EBSD data/Results/%s_IPFZ.png', file_names(file_number));
        saveas(gcf, file_name)
        if show_figures == false
            close(gcf)
        end
    end
    
    %% Plots the pole figures
    
    if plot_pf == true
        % plotting convention for pole figure
        setMTEXpref('xAxisDirection', 'north');
        setMTEXpref('zAxisDirection', 'intoplane');

        figure();
        ori = ebsd('Aluminium').orientations;
        x = [Miller(1, 0, 0, ori.CS), Miller(1, 1, 0, ori.CS), Miller(1, 1, 1, ori.CS)];
        plotPDF(ori, x, 'antipodal', 'contourf', 'minmax')
        mtexColorbar('FontSize', 25, 'Fontweight', 'bold');

        % set equal color range for all subplots
        setColorRange('equal')
        mtexColorbar ('location','southOutSide','title','mrd');

        % Turn off x and y labels on figures.
        pfAnnotations = @(varargin) [];
        setMTEXpref('pfAnnotations', pfAnnotations);

        % moving the vector3d axis labels outside of the hemisphere boundary
        text(vector3d.X,'RD','VerticalAlignment','bottom');
        text(vector3d.Y,'TD','HorizontalAlignment','left');

        % moving the hkil labels to make room for the rolling direction labels
        f = gcm; 
        f.children(1).Title.Position=[1, 1.25, 1];
        f.children(2).Title.Position=[1, 1.25, 1];
        f.children(3).Title.Position=[1, 1.25, 1];

        file_name = sprintf('EBSD data/Results/%s_pole_figure.png', file_names(file_number));
        saveas(gcf, file_name)
        if show_figures == false
            close(gcf)
        end
    end
    %% Plot the ODF sections
    if plot_odf == true
        figure();
        ori = ebsd('Aluminium').orientations;
        ori.SS = specimenSymmetry('orthorhombic');
        odf = calcDensity(ori);
        plot(odf, 'phi2', [0 45 65]* degree, 'antipodal', 'linewidth', 2, 'colorrange', [1 3.5]);
        ori1 = calcOrientations(odf, 2000);
        setColorRange('equal');
        mtexColorbar ('FontSize', 25, 'Fontweight', 'bold', 'location', 'south', 'title', 'mrd');
        file_name = sprintf('EBSD data/Results/%s_odf.png', file_names(file_number));
        saveas(gcf, file_name)
        if show_figures == false
            close(gcf)
        end
    end
end
