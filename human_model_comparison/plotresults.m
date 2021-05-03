N_TRIALS = 350;
subjs = ["01", '02', '04', '06', '09', '10', '12', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '35', '36', "37"];

f = figure;
ylim([0,1]);
hold on;

Y = zeros(numel(subjs), N_TRIALS);
for idx=1:numel(subjs)
    subj = subjs(idx);
    load(sprintf('Results_all/Sub%s/Result.mat', subj))
    
    y = Result(:,9) - 1;
    y(y == -1) = nan;
    Y(idx,:) = abs(y - 1);
end

contextchange = find(ischange(Result(:,2)));

text(10, 0.2, sprintf("%s", num2str(Result(1,2))))
for idx=1:numel(contextchange)
    trialidx = contextchange(idx);
    vline(trialidx)
    text(trialidx + 10, 0.2, sprintf("%s", num2str(Result(trialidx,2))))
end

M = zeros(numel(subjs), N_TRIALS);
for yidx=1:numel(subjs)
    for cidx=1:numel(contextchange)+1
        if cidx == 1
            s = 1;
        else
            s = contextchange(cidx-1);
        end
        
        if cidx == numel(contextchange)+1
            t = N_TRIALS;
        else
            t = contextchange(cidx);
        end

        M(yidx,s:t) = mean(Y(yidx,s:t),'omitnan');
    end
end

X = mean(M, 1, 'omitnan');
S = std(M, 1, 'omitnan') / sqrt(numel(subjs));
dg_plotShadeCL(gca, [1:N_TRIALS; X - S; X + S; X]', 'Color', 'b', 'LineWidth', 3);

plot(M', 'color', [0 0 1 0.1])

xlabel('Trial #')
ylabel('Performance')
legend('Human');
