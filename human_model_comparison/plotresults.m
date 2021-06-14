N_TRIALS = 350;
subjs = ["01", '02', '04', '06', '09', '10', '12', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '35', '36', "37"];

Y = zeros(numel(subjs), N_TRIALS);
for idx=1:numel(subjs)
    subj = subjs(idx);
    load(sprintf('Results_all/Sub%s/Result.mat', subj))
    
    y = Result(:,9) - 1;
    y(y == -1) = nan;
    Y(idx,:) = abs(y - 1);
end

model = load('seed0_vshuman_varblocks.mat');
P = model.performance(1001:4500);

%% Plot 0 style
f = figure;
ylim([0,1]);
hold on;

subplot(3,1,1);
plot_schedule()
X = mean(Y,1,'omitnan');
S = std(Y,1,'omitnan') / sqrt(numel(subjs));
dg_plotShadeCL(gca, [1:N_TRIALS; X - S; X + S; X]', 'Color', 'b', 'LineWidth', 2);
xlabel('Trial #')
ylabel('Average performance');
title('Window size = 1');

subplot(3,1,2);
plot_schedule()
X = movmean(mean(Y,1,'omitnan'),4);
S = movmean(std(Y,1,'omitnan') / sqrt(numel(subjs)),4);
dg_plotShadeCL(gca, [1:N_TRIALS; X - S; X + S; X]', 'Color', 'b', 'LineWidth', 2);
title('Window size = 4');

subplot(3,1,3);
plot_schedule()
X = movmean(mean(Y,1,'omitnan'),10);
S = movmean(std(Y,1,'omitnan') / sqrt(numel(subjs)),10);
dg_plotShadeCL(gca, [1:N_TRIALS; X - S; X + S; X]', 'Color', 'b', 'LineWidth', 2);
title('Window size = 10');


%% Plot 1b style

Pall = model.performance;
Pall_m = movmean(Pall,[100 0]);

figure;
hold on;
t = tiledlayout(1,1);
ax1 = axes(t);
plot_schedule()
ax1.XColor = 'b';
ax1.YColor = 'b';
plot_schedule()
X = movmean(mean(Y,1,'omitnan'),[10 0]);
S = movmean(std(Y,1,'omitnan') / sqrt(numel(subjs)),[10 0]);
dg_plotShadeCL(ax1, [1:N_TRIALS; X - S; X + S; X]', 'Color', 'b', 'LineWidth', 2);
xlabel('Trial #')
ylabel('Average performance -- Human');
ax2 = axes(t);
plot(ax2,Pall_m(1001:4500),'r')
ax2.XAxisLocation = 'top';
ax2.YAxisLocation = 'right';
ax2.Color = 'none';
ax1.Box = 'off';
ax2.Box = 'off';
xlim(ax2, [0 3500]);
ylim(ax2, [0 1])
ylabel('Average performance -- Model');
title('Model (window size = 100) vs Human (window size = 10)');

%% Regression
W_human = 15;
W_model = 150;

Ym = mean(Y,1,'omitnan');

xs = [];
ys = [];
erry = [];
errx = [];
for idx=1:N_TRIALS/W_human
    s_human = 1 + (idx-1) * W_human;
    t_human = s_human + W_human - 1;
    s_model = 1 + (idx-1) * W_model;
    t_model = s_model + W_model - 1;
    
    y = mean(Ym(s_human:t_human));
    x = mean(P(s_model:t_model));
    xs = [xs x];
    ys = [ys y];
    
    erry_ = std(Ym(s_human:t_human)) / sqrt(W_human);
    errx_ = std(P(s_model:t_model)) / sqrt(W_model);
    erry = [erry erry_];
    errx = [errx errx_];
end

[~,m1,b1] = regression(xs,ys);
fittedX = (min(xs)-0.01):0.01:(max(xs)+0.01);
fittedY = fittedX*m1+b1;
[R,Pv] = corrcoef(xs,ys); %P-value
if length(Pv) > 1
    pval = Pv(2);
    cor1 = R(2);
else
    pval = Pv;
    cor1 = R;
end

f = figure;
reward_str = ['Reward: Pearson Corr Coef = ' num2str(cor1) ' / Slope = ' num2str(m1) ' / P-Val = ' num2str(pval) ...
                newline 'W_{human} = ' num2str(W_human) ', W_{model} = ' num2str(W_model)];
hold on;
% errorbar(xs, ys, erry, 'color', 'k', 'LineStyle','none');
% errorbar(xs, ys, errx, 'horizontal', 'color', 'k', 'LineStyle','none');
scatter(xs, ys, 100, 'filled')
plot(fittedX,fittedY,'b','LineWidth',3);
xlabel('Model performance');
ylabel('Human performance');
title(reward_str)

%% Performance bars
h90 = mean([Y(:,1:40) Y(:,211:240)], 2, 'omitnan');
h30 = mean([Y(:,41:70) Y(:,241:280)], 2, 'omitnan');
h50 = mean([Y(:,71:110) Y(:,181:210)], 2, 'omitnan');
h70 = mean([Y(:,111:140) Y(:,281:320)], 2, 'omitnan');
h10 = mean([Y(:,141:180) Y(:,321:350)], 2, 'omitnan');

seeds = ["0", "7", "12", "57", "8", "1"];
Pall = zeros(size(seeds,2), 3500);
for idx=1:size(seeds,2)
    model_ = load(sprintf('seed%s_vshuman_varblocks.mat', seeds{idx}));
    Pall(idx,:) = model_.performance(1001:4500);
end

m90 = mean([Pall(:,101:400) Pall(:,2101:2400)], 2, 'omitnan');
m30 = mean([Pall(:,401:700) Pall(:,2401:2800)], 2, 'omitnan');
m50 = mean([Pall(:,701:1100) Pall(:,1801:2100)], 2, 'omitnan');
m70 = mean([Pall(:,1101:1400) Pall(:,2801:3200)], 2, 'omitnan');
m10 = mean([Pall(:,1401:1800) Pall(:,3201:3500)], 2, 'omitnan');

f = figure;
COLORS = cbrewer('qual', 'Set2', 10);
subplot(1,2,1);
plotbars({h90, h70, h50, h30, h10}, ["90%", "70%", "50%", "30%", "10%"], COLORS);
ylabel("Proportion of correct decisions");
title("Human performance")
subplot(1,2,2);
plotbars({m90, m70, m50, m30, m10}, ["90%", "70%", "50%", "30%", "10%"], COLORS);
title("Model performance")

p90 = ranksum(h90, m90);
fprintf("90 p-val=%0.5f \n", p90);
p70 = ranksum(h70, m70);
fprintf("70 p-val=%0.5f \n", p70);
p50 = ranksum(h50, m50);
fprintf("50 p-val=%0.5f \n", p50);
p30 = ranksum(h30, m30);
fprintf("30 p-val=%0.5f \n", p30);
p10 = ranksum(h10, m10);
fprintf("10 p-val=%0.5f \n", p10);

%% Performance
Ym = movmean(mean(Y,1,'omitnan'),15);
human50 = Ym(181:210);
human90 = Ym(211:240);
human30 = Ym(241:280);
human70 = Ym(281:320);
human10 = Ym(321:350);

Pm = movmean(P,150);
model50 = Pm(1801:2100);
model90 = Pm(2101:2400);
model30 = Pm(2401:2800);
model70 = Pm(2801:3200);
model10 = Pm(3201:3500);

f = figure;
subplot(1,2,1);
hold on;
plot(human50);
plot(human90);
plot(human30);
plot(human70);
plot(human10);
xlim([0 25]);
ylim([0.35 0.9]);
legend("50", "90", "30", "70", "10");
subplot(1,2,2);
hold on;
plot(model50);
plot(model90);
plot(model30);
plot(model70);
plot(model10);
xlim([0 250]);
ylim([0.35 0.9]);
legend("50", "90", "30", "70", "10");

%% Helpers
function plot_schedule()
load(sprintf('Results_all/Sub%s/Result.mat', "01"))
contextchange = find(ischange(Result(:,2)));
text(10, 0.2, sprintf("%s", num2str(Result(1,2))))
for idx=1:numel(contextchange)
    trialidx = contextchange(idx);
    vline(trialidx)
    text(trialidx + 10, 0.2, sprintf("%s", num2str(Result(trialidx,2))))
end
end

