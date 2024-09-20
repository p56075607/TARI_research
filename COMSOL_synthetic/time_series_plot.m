clear;clc;close all
% Import Python's pickle module
pickle = py.importlib.import_module('pickle');

% Open the file and read data using Python's open() function
fileID = py.open('median_RHOA_E1_and_date.pkl', 'rb');

% Load the first variable from the file
dates_E1 = pickle.load(fileID);

% Load the second variable from the file
median_RHOA_E1 = double(pickle.load(fileID));

% Close the file
fileID.close();

% 轉換 Python list (或 array) 為 MATLAB cell array
dates_E1_matlab = cellfun(@char, cell(dates_E1), 'UniformOutput', false);

% 將字串轉換為 MATLAB datetime 格式
dates_E1_datetime = datetime(dates_E1_matlab, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');

% 顯示結果
disp(dates_E1_datetime);
disp(median_RHOA_E1);

%%
% plot the time series
figure
plot(dates_E1_datetime, median_RHOA_E1, 'bo-','MarkerSize',5,'MarkerFaceColor','b');
xlabel('Date');
ylabel('Median RHOA');
title('Time Series of Median RHOA');
datetick('x', 'yyyy-mm-dd', 'keepticks');
grid on;
box on;
set(gca, 'FontSize', 14);
set(gcf, 'Color', 'w');
set(gcf, 'Position', [100, 100, 800, 500]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperOrientation', 'landscape');
