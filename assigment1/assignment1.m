%assigment 1
clc;
DATA3 = readtable('D:\LEARN\GRADUATE2020\IDA\assigment1\Biomechanical_Data_3Classes.csv');
DATA2 = readtable('D:\LEARN\GRADUATE2020\IDA\assigment1\Biomechanical_Data_2Classes.csv');
rdSelect = randi([1 230],1,230);
testSelect=randi([1 230],1,80);
data3=DATA3(rdSelect,:);
data2=DATA2(rdSelect,:);
testD2=data2(testSelect,:);
testD3=data3(testSelect,:);


%x=[pelvic_incidence	pelvic_tilt numeric	lumbar_lordosis_angle	sacral_slope	pelvic_radius	degree_spondylolisthesis];

DT2 = fitctree(data2, 'class');

%DT3 = fitctree(data3, 'class');

view(DT2,'mode','graph');
x=1:5;
a=[0.8625 0.85 0.8875 0.825 0.825];
pp=[0.92156863 0.92 0.90909091 0.87037037 0.87037037 ];
 pn=[0.75862069 0.73333333 0.84 0.73076923 0.73076923 ];
ra=[0.87037037 0.85185185 0.92592593 0.87037037 0.87037037 ];
rn=[0.84615385 0.84615385 0.80769231 0.73076923 0.73076923];
 