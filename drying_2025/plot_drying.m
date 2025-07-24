% plot_drying.m  使用 createfigure 繪圖
clear; close all; clc;
folder = fullfile(fileparts(mfilename('fullpath')), 'alpha_one_by_column_old');
files  = dir(fullfile(folder, 'dryingtime_alpha_one_RHOA_*_E1.csv'));
if isempty(files)
    error('未找到 CSV 檔，請確認路徑：%s', folder);
end

% 依 RHOA 編號排序
getNum = @(name) sscanf(name, 'dryingtime_alpha_one_RHOA_%d_E1.csv');
[~, idxSort] = sort(arrayfun(@(f) getNum(f.name), files));
files = files(idxSort);

Xc = {}; Yc = {}; names = {};
for k = 1:numel(files)
    T = readtable(fullfile(folder, files(k).name));
    if ~all(ismember({'delay_hours','median_RHOA'}, T.Properties.VariableNames))
        warning('%s 缺少必要欄位，跳過', files(k).name); continue; end
    Xc{end+1} = T.delay_hours; %#ok<*AGROW>
    Yc{end+1} = T.median_RHOA;
    token = regexp(files(k).name, 'RHOA_(\d+)_', 'tokens', 'once');
    if ~isempty(token)
        names{end+1} = sprintf('RHOA %s', token{1});
    else
        names{end+1} = files(k).name;
    end
end

% 呼叫通用繪圖函式
createfigure(Xc, Yc, names);
