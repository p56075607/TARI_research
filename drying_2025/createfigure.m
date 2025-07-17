function createfigure(Xcell, Ycell, legend_names)
%CREATEFIGURE 通用半對數繪圖函式
%   Xcell, Ycell : 1×N cell,  每 cell 為一條曲線的 x 與 y 資料
%   legend_names : 1×N cell   圖例文字
% 範例: createfigure({x1,x2},{y1,y2},{'A','B'})

if nargin < 2
    error('至少需要 Xcell 及 Ycell 兩個參數');
end
if nargin < 3 || isempty(legend_names)
    legend_names = arrayfun(@(k) sprintf('Series %d',k), 1:numel(Xcell), 'uni',0);
end

% 預設顏色、標記
colors  = lines(numel(Xcell));
markers = {'o','s','^','d','v','>','<','p','h','x','+','*','.'};

figure('Position',[100 100 1400 800]); hold on;
for k = 1:numel(Xcell)
    mk = markers{ mod(k-1, numel(markers))+1 };
    semilogy(Xcell{k}, Ycell{k}, mk, 'Color', colors(k,:), 'LineWidth',1.5,...
             'MarkerSize',4, 'DisplayName', legend_names{k});
end

ylabel('Apparent Resistivity (Ω·m)','FontWeight','bold');
xlabel('Delay Hours','FontWeight','bold');

title('Alpha-one RHOA vs. Delay Hours (多曲線)','FontWeight','bold');

grid on; box on;
set(gca,'FontSize',12,'LineWidth',1.5,'YScale','log');
legend('Location','eastoutside');
end

