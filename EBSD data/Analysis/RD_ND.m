%% Import Script for EBSD Data
%
% This script was automatically created by the import wizard. You should
% run the whoole script or parts of it in order to import your data. There
% is no problem in making any changes to this script.

%% Specify Crystal and Specimen Symmetries

% crystal symmetry
CS = {... 
  'notIndexed',...
  crystalSymmetry('m-3m', [4.05 4.05 4.05], 'mineral', 'Aluminium', 'color', 'light blue')};

% plotting convention
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','outofplane');

%% Specify File Names

% path to files
pname = 'H:\Surfalex_All data\EBSD data\Data\RD-ND plane';

% which files to be imported
fname = [pname '\RD-ND Data.ctf'];

%% Import the Data

% create an EBSD variable containing the data
ebsd = loadEBSD(fname,CS,'interface','ctf',...
  'convertEuler2SpatialReferenceFrame');

rot = rotation('axis', xvector, 'angle',90*degree);
ebsd = rotate(ebsd, rot);

ebsd = ebsd('indexed');
[grains, ebsd.grainId] = calcGrains(ebsd,'angle',5*degree);
grains = smooth(grains);

plot(ebsd,ebsd.bc);
colormap gray
mtexColorbar;

hold on

oM = ipfHSVKey(ebsd('indexed'))
oM.inversePoleFigureDirection = xvector;
color = oM.orientation2color(ebsd('indexed').orientations);
plot(ebsd('indexed'),color);

hold off

figure ()

ori=ebsd('Aluminium').orientations
x=[Miller(1,1,1,ori.CS),Miller(2,0,0,ori.CS),Miller(2,2,0,ori.CS)]; % include hkil figures here
plotPDF(ori,x,'antipodal','contourf','colorrange',[1 3.5])
colorbar;
