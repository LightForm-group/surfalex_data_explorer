% Run on Matlab R2020a with MTEX 5.7.

% Before begining analysis change the working directory to the "analysis"
% folder of the surfalex_data_explorer.

data_dir = '../data/texture_analysis';
results_dir = '../results/';
zip_path = [data_dir '/ebsd_maps.zip'];
zenodo_URL = "https://sandbox.zenodo.org/record/811311/files/ebsd.zip";

% Check if there is a data folder
if ~exist(data_dir, 'dir')
    mkdir(data_dir)
end

% Check if there is a results folder
if ~exist(results_dir , 'dir')
    mkdir(results_dir )
end

% Check if there is already data and download it if not
if ~exist(zip_path, 'file')
    urlwrite(zenodo_URL, zip_path);
end

% unzip data files.
filenames = unzip(zip_path, data_dir);
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

MAP_X_DIMS = [5600 6713 5045];
MAP_PATCH_MARGIN_Y = [70 70 170];

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
    fname = sprintf("%s/%s.ctf", data_dir, file_names(file_number));
    ebsd = EBSD.load(fname, CS, 'interface', 'ctf', 'convertEuler2SpatialReferenceFrame');

    % Rotate the map. Rotation angle depends on which plane is being considered
    phi = rotations(file_number * 3 - 2);
    theta = rotations(file_number * 3 - 1 );
    psi = rotations(file_number * 3);
    ebsd = rotate(ebsd, rotation('euler', phi * degree, theta * degree, psi * degree), 'keepXY');

    %% Plot the Band contrast and IPFZ maps together.
    if plot_ipf == true
        
        setMTEXpref('FontSize', 10);
        
        % Calculate grains from map
        ebsd = ebsd('indexed');
        [grains, ebsd.grainId] = calcGrains(ebsd, 'angle', 5 * degree);
        grains = smooth(grains);

        % Plotting convention for IPF map
        setMTEXpref('xAxisDirection','east');
        setMTEXpref('zAxisDirection','intoplane');

        % Plot band contrast to show grain morphology:
        figure();
        [~,mP] = plot(ebsd, ebsd.bc);
        mP.micronBar.length = 750;
        colormap gray
        file_name = sprintf('%s/band_contrast_%s.png', results_dir, file_names(file_number));
        exportgraphics(gcf,file_name);        
        if show_figures == false
            close(gcf)
        end
                
        % Plot IPF map:               
        oM = ipfHSVKey(ebsd('indexed'));
        oM.inversePoleFigureDirection = zvector;
        color = oM.orientation2color(ebsd('indexed').orientations);
        fig = figure();
        [h,mP] = plot(ebsd('indexed'),color);        
        mP.parent = fig;
        mP.micronBar.length = 750;        
                
        % Add RD/TD/ND label as a patch:
        patch_width = 750 * MAP_X_DIMS(file_number) / MAP_X_DIMS(1);
        patch_x_margin = 70 * MAP_X_DIMS(file_number) / MAP_X_DIMS(1);
        patch_y_margin = MAP_PATCH_MARGIN_Y(file_number);
        patch_height = 270 * MAP_X_DIMS(file_number) / MAP_X_DIMS(1);                
        dir_patch = patch([MAP_X_DIMS(file_number) - patch_x_margin  - patch_width...
                MAP_X_DIMS(file_number) - patch_x_margin ...
                MAP_X_DIMS(file_number) - patch_x_margin ...
                MAP_X_DIMS(file_number) - patch_x_margin  - patch_width...
            ],...
            [patch_y_margin patch_y_margin...
            patch_y_margin + patch_height...
            patch_y_margin + patch_height],...
            'y');                        
        txt_y_pos = (patch_y_margin + (patch_height / 2)) * 0.93;
        text(MAP_X_DIMS(file_number) - patch_x_margin - (patch_width / 2),...
            txt_y_pos,...
            file_names(file_number),...
            'HorizontalAlignment', 'center',...
            'VerticalAlignment', 'middle',...
            'color', 'w',...
            'fontSize', 12);                
        set(dir_patch, 'FaceColor', 'k', 'EdgeColor', 'none', ...
            'LineWidth', 1, 'FaceAlpha', 0.6);        
        
        set(gcf,'Position',[100 100 800 800])
        mP.ax.Units = 'centimeter';
        mP.ax.Position(3) = 12;
        
        file_name = sprintf('%s/IPFZ_%s.png', results_dir, file_names(file_number));        
        exportgraphics(gcf,file_name,'Resolution',300);

        if show_figures == false
            close(gcf)
        end   
        
        if file_number == 1
            % Plot the IPF colour key:
            setMTEXpref('FontSize', 25);
            fig = figure();
            plot(oM);
            file_name = [results_dir '/IPFZ_colour_key.png'];
            exportgraphics(gcf,file_name);
            if show_figures == false
                close(gcf)
            end
        end
        setMTEXpref('FontSize', 20);        
    end
    
    %% Plots the pole figures
    
    if plot_pf == true
        % plotting convention for pole figure
        setMTEXpref('xAxisDirection', 'north');
        setMTEXpref('zAxisDirection', 'intoplane');

        figure();
        ori = ebsd('Aluminium').orientations;
        x = [Miller(1, 0, 0, ori.CS), Miller(1, 1, 0, ori.CS), Miller(1, 1, 1, ori.CS)];

        plotPDF(ori, x, 'antipodal', 'contourf', 'earea', 'LineWidth', 0.1);

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

        file_name = sprintf('%s/pole_figure_%s.pdf', results_dir, file_names(file_number));
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
            file_name = sprintf('%s/pole_figure_cbar.pdf', results_dir);
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
        file_name = sprintf('%s/ODF_%s.png', results_dir, file_names(file_number));
        saveas(gcf, file_name)
        if show_figures == false
            close(gcf)
        end
    end
end
