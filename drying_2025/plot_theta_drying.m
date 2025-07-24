%% plot_theta_drying.m
% 讀取 theta_rho.csv 中 mean_1m 含水量並繪製時序圖，
% 並根據 E1_window.csv 的開始與結束時間以垂直線標示。
% Author: Auto-generated

%% 參數設定
clear; clc;

baseDir = fileparts(mfilename('fullpath'));
thetaFile = fullfile(baseDir, 'theta_rho.csv');
windowFile = fullfile(baseDir, 'E1_window.csv');

%% 讀取含水量資料
T = readtable(thetaFile, 'Delimiter', ',');
% 將 date_time 轉為 datetime
T.date_time = datetime(T.date_time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

%% 讀取事件窗口
W = readtable(windowFile, 'Delimiter', ',');
W.x = datetime(W.x, 'InputFormat', 'yyyy/MM/dd HH:mm');

%% 繪圖
figure('Color', 'w', 'Position', [100 100 1000 400]);
plot(T.date_time, T.mean_1m, 'b-', 'LineWidth', 1.2); hold on;

% 垂直線 (每列 x 視為需要標示的時間點)
for i = 1:height(W)
    xline(W.x(i), '--r', 'LineWidth', 1);
end

xlabel('Datetime');
ylabel('mean\_1m (Vol. Water Content)');
title('mean\_1m 含水量時序圖');
grid on;
datetikformat = 'yyyy-MM-dd';
xtickformat(datetikformat);

legend({'mean\_1m', 'Event Window'}, 'Location', 'best');

%% 儲存圖檔
outPNG = fullfile(baseDir, 'mean_1m_theta_drying.png');
saveas(gcf, outPNG);
fprintf('圖形已儲存至 %s\n', outPNG);
