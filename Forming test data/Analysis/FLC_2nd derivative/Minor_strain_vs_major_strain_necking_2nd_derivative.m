clc;
clear;

%provide the sheet number corresponding to which major and minor strain
%data till necking is required

%sheet1=10 mm
%sheet2= 20 mm
%sheet3= 40 mm
%sheet4 = 60 mm
%sheet5 = 120 mm
%sheet6 = full circle

raw = xlsread('H:\Surfalex_All data\Forming test data\Analysis\FLC_last stage before crack\Strain path data_time.xlsx','Sheet6','A4:W200');


time=[1, 9, 17]; %this is because time is always provided in 1st, 9th and 17th column of each sheet.
major_strain=[3,5,7,11,13,15,19,21,23]; ; %this is because major strain is always provided in 3rd, 5th, 7th and so on columns in the excel sheet.
minor_strain=[2,4,6,10,12,14,18,20,22];


for ii=1:size(major_strain,2)
    
collected_Major_Strain=[]; 
collected_Minor_Strain=[];

y=raw(:, major_strain(ii));

if ii<=3

x=raw(:,time (1));

end

if ii>3 && ii<=6

x=raw(:,time (2));

end

if ii>6 && ii<=9

x=raw(:,time (3));

end

%remove NAN from the column if any
x(~any(~isnan(x), 2),:)=[];
y(~any(~isnan(y), 2),:)=[];

%Fit a polynomial with experimental data and verify the fit here

p=polyfit(x, y, 15);
x1=linspace(0,max(x),500);
y1=polyval(p,x1);
% % plot(x,y,'o')
% hold on
% % plot(x1,y1)
% hold off
% title('polynomial fit with experimental data'); 
% figure ()

%Calculate the first differential here and apply some somoothening
dx = mean(diff(x1));                                
dy = gradient(y1,dx);
% % scatter(x1,dy);
% hold on
dy1 = smoothdata(dy,'sgolay',20);
% plot(x1,dy1);
% title('1st differential in scatter and smoothened curve in line'); 
% figure ()

%Calculate the 2nd derivative here
dy2=gradient(dy1,dx);
% % scatter(x1,dy2);
% title('2nd differential in scatter'); 
% figure ()

%determined the necking criteria as 10% of the maximum of 2nd derivative
neck_criteria= (0.1*max(dy2));

%determining the time corresponding to the necking
A=(dy2-neck_criteria);
l = find(A > 0,1);
actual_time_at_neck= x1(l);

%match the time obatined from here with the raw time data and then select
%only the major and minor strain data till that time

B=(x-actual_time_at_neck);
m=find(B>0,1);



for i=1:m

     collected_Major_Strain=[collected_Major_Strain; raw(i,major_strain(ii))];
     collected_Minor_Strain=[collected_Minor_Strain;raw(i,minor_strain(ii))];
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
%'Strain path data_2nd derivative.xlsx'



