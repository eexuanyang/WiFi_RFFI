close all; 
clear all; 
clc

long_subca = [7:32 34:59];

legd_size_large = 25;
gca_size_large = 27;

%% Channel
load("lltf_fre_tx1_rx1_channel.mat")
tx1_rx1_channel = abs(lltf_fre);
load("lltf_fre_tx2_rx1_channel.mat")
tx2_rx1_channel = abs(lltf_fre);
load("lltf_fre_tx3_rx1_channel.mat")
tx3_rx1_channel = abs(lltf_fre);
load("lltf_fre_tx1_rx2_channel.mat")
tx1_rx2_channel = abs(lltf_fre);
load("lltf_fre_tx2_rx2_channel.mat")
tx2_rx2_channel = abs(lltf_fre);
load("lltf_fre_tx3_rx2_channel.mat")
tx3_rx2_channel = abs(lltf_fre);

tx1_rx1_channel = tx1_rx1_channel(long_subca);
tx2_rx1_channel = tx2_rx1_channel(long_subca);
tx3_rx1_channel = tx3_rx1_channel(long_subca);
tx1_rx2_channel = tx1_rx2_channel(long_subca);
tx2_rx2_channel = tx2_rx2_channel(long_subca);
tx3_rx2_channel = tx3_rx2_channel(long_subca);

data = [tx1_rx1_channel; tx2_rx1_channel; tx3_rx1_channel; tx1_rx2_channel; tx2_rx2_channel; tx3_rx2_channel];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);

ax = gca;
ax.Units = 'normalized';  
ax.Position = [0.15, 0.3, 0.5, 0.4];  

ax.DataAspectRatio = [1, 1, 1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot(tx2_rx1_channel,'-','Color',[0.4660 0.6740 0.1880],LineWidth=3)
hold on
plot(tx2_rx2_channel,'r-.',LineWidth=3)
hold on
plot(tx3_rx1_channel,'b--',LineWidth=3)
hold on
plot(tx3_rx2_channel,'k:',LineWidth=3)
% legend("T1-R1","T1-R2","T2-R1","T2-R2")
lgd = legend("T1-R1","T1-R2","T2-R1","T2-R2", ...
    'FontSize', legd_size_large, ...
    'NumColumns', 2);

xlabel({'Subcarrier', '(a)'})
ylabel("Amplitude")
xlim([0 53])
ylim([min(data) max(data)])
xticks([0 10 20 30 40 50])
set(gca,'FontSize',gca_size_large,'FontName', 'Times New Roman');
 
%% RFF Feature
load("lltf_fre_tx1_rx1_filter.mat")
tx1_rx1_data = abs(lltf_fre);
load("lltf_fre_tx2_rx1_filter.mat")
tx2_rx1_data = abs(lltf_fre);
load("lltf_fre_tx3_rx1_filter.mat")
tx3_rx1_data = abs(lltf_fre);
load("lltf_fre_tx1_rx2_filter.mat")
tx1_rx2_data = abs(lltf_fre);
load("lltf_fre_tx2_rx2_filter.mat")
tx2_rx2_data = abs(lltf_fre);
load("lltf_fre_tx3_rx2_filter.mat")
tx3_rx2_data = abs(lltf_fre);

tx2_rx1_rff = tx2_rx1_data(long_subca) ./ tx1_rx1_data(long_subca);
tx2_rx2_rff = tx2_rx2_data(long_subca) ./ tx1_rx2_data(long_subca);
tx3_rx1_rff = tx3_rx1_data(long_subca) ./ tx1_rx1_data(long_subca);
tx3_rx2_rff = tx3_rx2_data(long_subca) ./ tx1_rx2_data(long_subca);

tx2_rx1_rff = tx2_rx1_rff ./ sqrt(sum(abs(tx2_rx1_rff).^2));
tx2_rx2_rff = tx2_rx2_rff ./ sqrt(sum(abs(tx2_rx2_rff).^2));
tx3_rx1_rff = tx3_rx1_rff ./ sqrt(sum(abs(tx3_rx1_rff).^2));
tx3_rx2_rff = tx3_rx2_rff ./ sqrt(sum(abs(tx3_rx2_rff).^2));

data = [tx2_rx1_rff; tx2_rx2_rff; tx3_rx1_rff; tx3_rx2_rff];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);

ax = gca;
ax.Units = 'normalized';  
ax.Position = [0.15, 0.3, 0.5, 0.4]; 

ax.DataAspectRatio = [1, 1, 1];  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot(tx2_rx1_rff,'-','Color',[0.4660 0.6740 0.1880],LineWidth=3)
hold on
plot(tx2_rx2_rff,'r-.',LineWidth=3)
hold on
plot(tx3_rx1_rff,'b--',LineWidth=3)
hold on
plot(tx3_rx2_rff,'k:',LineWidth=3)
% legend("T1-R1","T1-R2","T2-R1","T2-R2")
lgd = legend("T1-R1","T1-R2","T2-R1","T2-R2", ...
    'FontSize', legd_size_large, ...
    'NumColumns', 2);

xlabel({'Subcarrier', '(b)'})
ylabel("Amplitude")
xlim([0 53])
ylim([min(data) max(data)])
xticks([0 10 20 30 40 50])
set(gca,'FontSize',gca_size_large,'FontName', 'Times New Roman');