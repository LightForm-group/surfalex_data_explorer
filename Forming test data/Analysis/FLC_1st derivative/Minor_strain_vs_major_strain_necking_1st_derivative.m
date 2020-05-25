clc;
clear all;


raw = xlsread('H:\Surfalex_All data\Forming test data\Analysis\FLC_last stage before crack\Strain path data_time.xlsx','Sheet1','A4:W200');
raw1= xlsread('H:\Surfalex_All data\Forming test data\Analysis\FLC_1st derivative\Strain path data_away from neck.xlsx','Sheet1','A4:W200');

time=[1, 9, 17];
major_strain=[3,5,7,11,13,15,19,21,23];
minor_strain=[2,4,6,10,12,14,18,20,22];

 for ii=1:size(major_strain,2)
   
collected_Major_Strain=[]; 
collected_Minor_Strain=[];    
     
     
if ii<=3

x=raw(:,time (1));

end

if ii>3 && ii<=6

x=raw(:,time (2));

end

if ii>6 && ii<=9

x=raw(:,time (3));

end   
     
% read the major strain versus time data in the crack/neck region
y=raw(:, major_strain(ii));
% read the major strain versus time data away from the crack/neck region
y1=raw1(:, major_strain(ii));

%remove all NAN's
x(~any(~isnan(x), 2),:)=[];
y(~any(~isnan(y), 2),:)=[];
y1(~any(~isnan(y1), 2),:)=[];

scatter (x,y);
hold on
scatter(x,y1);

figure ();

%Do a polynomial fit for the major strain versus time data away from the crack/neck region
p=polyfit(x, y1, 15);
x1=linspace(0,max(x),500);
y2=polyval(p,x1);

%Select the major strain data after 20 seconds as there is lot of
%fluctuations in the inital data

x1=x1(300:end);
y2=y2(300:end);
scatter(x1,y2);

figure()

%calculate the slope change to see identify the region close to necking

[TF,~,~] = ischange(y2,'linear','MaxNumChanges',2);
cpts = find(TF);

scatter(x1, y2)
hold on
plot(x1(cpts), y2(cpts), '^r', 'MarkerFaceColor','r')
hold off

figure ()

%Reshape the major strain data near the neck region and find the first
%slope change. Use this as the criteria for necking

if size(cpts,2)==2
    y2=y2(cpts(2):(end));
    x1=x1(cpts(2):(end));
else
    y2=y2(cpts(1):(end));
    x1=x1(cpts(1):(end)); 
end
    


[TF1,~,~] = ischange(y2,'linear','MaxNumChanges',2);
cpts1 = find(TF1);

scatter(x1, y2)
hold on
plot(x1(cpts1), y2(cpts1), '^r', 'MarkerFaceColor','r')
hold off

index=x1(cpts1(1));

A= index/max(x1);

if A>0.968
    
    index=0.94*max(x1);
end

if ii == 2 || ii==3 || ii==5 || ii==6 || ii==8 || ii==9 
    
    for i=1:m

     collected_Major_Strain=[collected_Major_Strain; raw(i,major_strain(ii))];
     collected_Minor_Strain=[collected_Minor_Strain;raw(i,minor_strain(ii))];
   end

else
    B=(x-index);
    m = find(B > 0,1);
 
    for i=1:m

     collected_Major_Strain=[collected_Major_Strain; raw(i,major_strain(ii))];
     collected_Minor_Strain=[collected_Minor_Strain;raw(i,minor_strain(ii))];
   end
end


final{2*ii+1} = num2cell(collected_Major_Strain);
final{2*ii}= num2cell(collected_Minor_Strain);
end

col = size(final,2);
row = cellfun('size',final,1);
out = cell(max(row),col);
for k = 1:col
    out(1:row(k),k) = final{k};
end

xlswrite('filename.xlsx',out,'sheet1','A2');
xlswrite('filename.xlsx',{'S1_Sec1_minor','S1_Sec1_major','S1_Sec2_minor','S1_Sec2_major','S1_Sec0_minor','S1_Sec0_major','S2_Sec1_minor','S2_Sec1_major','S2_Sec2_minor','S2_Sec2_major','S1_Sec0_minor','S2_Sec0_major','S3_Sec1_minor','S3_Sec1_major','S3_Sec2_minor','S3_Sec2_major','S3_Sec0_minor','S3_Sec0_major'},'sheet1','B1');

close all;


% Specimen names: S1_sec1_minor= specimen001_section_one_minor strain
%                 S1_sec1_major= specimen001_section_one_major strain

%                 S1_sec2_minor= specimen001_section_two_minor strain
%                 S1_sec2_major= specimen001_section_two_major strain

%                 S1_sec0_minor= specimen001_section_zero_minor strain
%                 S1_sec0_major= specimen001_section_zero_major strain

%                 S2_sec1_minor= specimen002_section_one_minor strain
%                 S2_sec1_major= specimen002_section_one_major strain

%                  And so on. 

%For running the script, change the sheet number from one to six in line 4
%sheet1=10 mm geometry
%sheet2=20 mm geometry
%sheet3=40 mm geometry
%sheet4=60 mm geometry
%sheet5=120 mm geometry
%sheet6=full circle 177 mm geometry

% Once the script is executed a excel file named as 'filename.xlsx' will
% appear in the folder. 

%This excel sheet will have the major and minor strain data till necking
%for all the sections and the repeat specimens for a particluar geometry

% the abbreviation of the column labels is provided in line numbers
% 106-118.


% once the script is executed for all the test geometries, the major and
% minor strain data is condensed in a single excel file 
%'Strain path data_first derivative.xlsx'
