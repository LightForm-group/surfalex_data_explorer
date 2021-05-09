% Before begining analysis change the working directory to the EBSD folder
% of the surfalex_data_explorer.

% Check if there is a data folder
if ~exist('./data', 'dir')
    mkdir('data')
end

% Check if there is a results folder
if ~exist('./results', 'dir')
    mkdir('results')
end

% Check if there is already data and download it if not
if ~exist("data/ebsd_maps.zip", 'file')
    urlwrite("https://sandbox.zenodo.org/record/811311/files/ebsd.zip", "data/ebsd_maps.zip")
end

% unzip data files.
filenames = unzip("data/ebsd_maps.zip", "data");
fprintf("\nData sucessfully downloaded and unzipped\n")

%% EBSD figures for Surfalex material using MTEX
% Specify Crystal Symmetries

CS = {... 
  'notIndexed',...
  crystalSymmetry('m-3m', [4.05 4.05 4.05], 'mineral', 'Aluminium', 'color', 'light blue')};
setMTEXpref('defaultColorMap',parula);

%% Specify File Names

file_names = ["RD-ND", "RD-TD", "TD-ND"];
pf_xy_labels = ["RD", "ND"; "RD", "TD"; "TD", "ND"];
rotations = [[0, -90, 0], [-90, 0, 0], [0, 90, 0]];

plot_ipf = true;
plot_pf = true;
plot_odf = true;
show_figures = false;

%% Set figure preferences

setMTEXpref('FontSize', 20);
setMTEXpref('FontName', 'SansSerif')
setMTEXpref('figSize', 'small');
setMTEXpref('outerPlotSpacing', 60);
setMTEXpref('innerPlotSpacing', 20)

PF_cbar_limits_global = [0.2 3.9];

for file_number = 1:length(file_names)
    % create an EBSD variable containing the data
    fname = sprintf("./data/%s.ctf", file_names(file_number));
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
        file_name = sprintf('./results/%s_IPFZ.png', file_names(file_number));
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

        plotPDF(ori, x, 'antipodal', 'contourf', 'earea', 'LineWidth', 0.3);

        % Turn off x and y labels on figures.
        pfAnnotations = @(varargin) [];
        setMTEXpref('pfAnnotations', pfAnnotations);

        f = gcm;
        setColorRange('equal'); % Use the same colour range for all sub-figures:
        mtexColorbar();
        f.colorbar; % Turn off colour bar
        caxis(PF_cbar_limits_global);
                    
        % RD/TD/ND labels:
        text(0, 1.6, 0, char(pf_xy_labels(file_number, 1)),'HorizontalAlignment','center','fontSize',22);
        text(1.5, 0, 0, char(pf_xy_labels(file_number, 2)),'VerticalAlignment','middle','fontSize',22);

        % moving the hkil labels to make room for the rolling direction labels
        f.children(1).Title.Position=[1, 1.25, 1];
        f.children(2).Title.Position=[1, 1.25, 1];
        f.children(3).Title.Position=[1, 1.25, 1];

        file_name = sprintf('./results/pole_figure_%s.pdf', file_names(file_number));
        exportgraphics(gcf,file_name,'ContentType','vector');
        if show_figures == false
            close(gcf)
        end
        
        % Generate a separate colour bar:
        if file_number == 1
            figure();
            axis off;
            cb = colorbar();
            caxis(PF_cbar_limits_global);
            cb.FontSize = 15;
            cb.Location = 'south';
            cb.Ruler.TickLabelFormat = '%.1f';        
            set(get(cb, 'Title'),'String','MRD');
            file_name = sprintf('./results/pole_figure_cbar.pdf');
            exportgraphics(gcf,file_name,'ContentType','vector');
            if show_figures == false
                close(gcf)
            end
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
        file_name = sprintf('./results/%s_odf.png', file_names(file_number));
        saveas(gcf, file_name)
        if show_figures == false
            close(gcf)
        end
    end
end
