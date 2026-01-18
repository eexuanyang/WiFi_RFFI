clear all
close all
clc

load("low_freq_score.mat")
load("class_acc_UseEachDevAsRefArr_V1_model_sig.mat")

class_ave_arr = zeros(5, 11);
for row_idx = 1:5
    class_ave_arr(row_idx, :) = mean(class_acc_UseEachDevAsRefArr_V1_model_sig(1+(row_idx-1)*4:4*row_idx, :));
end

low_freq_score_vec = low_freq_score(:);
class_acc_result_use_each_dev_as_ref_arr_vec = class_ave_arr(:);

X = low_freq_score_vec;
Y = class_acc_result_use_each_dev_as_ref_arr_vec;
[r,p_value] = corr(X, Y);

figure;
scatter(X, Y, 50, 'filled', 'MarkerFaceAlpha', 0.6, DisplayName='Samples');
hold on;
coefficients = polyfit(X, Y, 1);
x_fit = linspace(min(X), max(X), 100);
y_fit = polyval(coefficients, x_fit);
plot(x_fit, y_fit, 'LineWidth', 3, DisplayName='LS Fitting');
legend('Location', 'best', 'FontSize', 23); %
xlim([0.87 1])
ylim([0.35  1])
% xlabel({'Pearson Correlation Coefficient \it{r}'},'FontName', 'Times New Roman');
xlabel({'{\it{\eta}}_{LF}'},'FontName', 'Times New Roman');
yticks(0:0.2:1)
ylabel('Classification Accuracy','FontName', 'Times New Roman');
title(['{\it{ p}}-value = ' num2str(round(p_value,4))])
set(gca,'FontSize',23,'FontName', 'Times New Roman');
box on;  % 