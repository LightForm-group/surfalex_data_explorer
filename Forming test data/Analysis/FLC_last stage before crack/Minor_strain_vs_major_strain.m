clc;
clear;

collected_Major_Strain=[]; % Initialization of blank matrix for future use
collected_Minor_Strain=[]; % Initialization of blank matrix for future use

%Define the path here to export the major and minor strain data % Just

%change the sample geometry and section 

%e.g Surfalex_20mm_001\Section zero\major strain
%    Surfalex_20mm_001\Section zero\major strain

%    Surfalex_20mm_001\Section one\major strain
%    Surfalex_20mm_001\Section one\major strain

%    Surfalex_20mm_001\Section two\major strain
%    Surfalex_20mm_001\Section two\major strain

files_major = dir(['H:\Surfalex_All data\Forming test data\Data\Strain data_All stages\Surfalex_10mm_001\Section zero\major strain\', '\*.csv']);
files_minor = dir(['H:\Surfalex_All data\Forming test data\Data\Strain data_All stages\Surfalex_10mm_001\Section zero\minor strain\', '\*.csv']);

%store fullname for major strain

for ii = length(files_major):-1:1 
      % Create the full file name and partial filename
      path= files_major.folder;
      pathname = string(path);
      fullname1(ii,1) = strcat(pathname,'\',files_major(ii).name);   
end

%store fullname for minor strain
for jj = length(files_minor):-1:1 
      % Create the full file name and partial filename
      path= files_minor.folder;
      pathname = string(path);
      fullname2(jj,1) = strcat(pathname,'\',files_minor(jj).name);   
end

%Please provide the stage number for the last stage before crack appears in
%the sample. For each sample, this data is provided in the read me ppt (slide 21).

prompt = 'Enter the file number for the last stage before crack(''provided in the read me file''):-';
stage_number = input(prompt);

%Display the last stage before crack appears in the sample
disp('Detected File for last stage before crack appears in the sample')
disp(files_major(stage_number).name)

%Read the last file here using readtable function'
T = readtable(fullname1(stage_number),'Delimiter', ';', 'HeaderLines',6);
T = rmmissing(T); % It removes missing entries from an array (to remove NaN)
height_data = size(T(:,6)); %this is to estimate the size of the 6th column as  it corresponds to major strain

%Making sure that the all the data are in numerics. Data should not be in
%string or cell

if iscell(T(1,6).Var6)==0
  T = table2array(T);   
else

for i = 1:height_data(1)
    strain_col(i,1) = str2num(cell2mat(T(i,6).Var6));
end
T = [T.Var1,T.Var2,T.Var3,T.Var4,T.Var5,strain_col];
end

%Determining the maximum strain and its location: 
%It is essential to find the maxima as well as it position. Once the position is determined, the major and minor strain values
%corresponding to this position can be extracted for all the data files viz. from 'Section1_0001.csv' to 'Section1_0143.csv'

[~,I] = max(T(:,6));

plot (T(:,1),T(:,6),'linewidth',3);
xlabel('index','fontweight','bold','fontsize',32)
ylabel('Major strain','fontweight','bold','fontsize',32)

%% Reading all the Files having Major strain for corresponding ID
% Extension of File containing the Major strain should be *.csv

for k = 1:stage_number

strain_from_files = readtable(fullname1(k),'Delimiter', ';', 'HeaderLines',6);
strain_from_files = rmmissing(strain_from_files);
if iscell(strain_from_files(1,6).Var6)==0
    strain_from_files = table2array(strain_from_files);
else
    
height_data = size(strain_from_files(:,6));
for i = 1:height_data(1)
    strain_col(i,1) = str2num(cell2mat(strain_from_files(i,6).Var6));
end
strain_from_files = [strain_from_files.Var1,strain_from_files.Var2,strain_from_files.Var3,strain_from_files.Var4,strain_from_files.Var5,strain_col];
end
Detected_strain = strain_from_files(I,6);
collected_Major_Strain = [collected_Major_Strain, Detected_strain];
end

%Reading all the Files having Minor strain strain for corresponding ID

for i = 1:stage_number

strain_from_files = readtable(fullname2(i),'Delimiter', ';', 'HeaderLines',6);
strain_from_files = rmmissing(strain_from_files);
if iscell(strain_from_files(1,6).Var6)==0
    strain_from_files = table2array(strain_from_files);
else
height_data = size(strain_from_files(:,6));
for i = 1:height_data(1)
    strain_col(i,1) = str2num(cell2mat(strain_from_files(i,6).Var6));
end
strain_from_files = [strain_from_files.Var1,strain_from_files.Var2,strain_from_files.Var3,strain_from_files.Var4,strain_from_files.Var5,strain_col]
end

Detected_strain = strain_from_files(I,6);
collected_Minor_Strain = [collected_Minor_Strain, Detected_strain];
end

minor_strain=[collected_Minor_Strain]';
major_strain=[collected_Major_Strain]';

strain_path=[minor_strain, major_strain];

plot(collected_Minor_Strain,collected_Major_Strain,'-o','MarkerSize',3);
xlabel('Minor Strain')
ylabel('Major Strain')

xlswrite('file.xlsx',strain_path,'sheet1','A2');
xlswrite('file.xlsx',{'minor strain','major strain'},'sheet1','A1');