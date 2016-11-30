function unt()

%clear all;
close all;
set(gcf,'color','w');

iscenario = 1;
colors = {'red','blue','green','black','magenta','cyan', ...
    [0.4 0.7 0.1],[0.7 0.4 0.1],[0.1 0.4 0.7],[0.7, 0.7, 0]};


if (iscenario == 1)
    plotY = 1; % 1 - plot train_loss, 2 - plot valid_loss, 3 - plot valid_err
    if (plotY == 1)    columnidx = 3;  ylabeltext = 'Training loss';   end;
    if (plotY == 2)    columnidx = 4;  ylabeltext = 'Validation loss';   end;
    if (plotY == 3)    columnidx = 5;  ylabeltext = 'Validation error (%)';   end;
    for irun = [1, 2, 3]
      	if (irun == 1)  leg = {};                               end;
        if (irun == 2)  legend(leg,'Location','NorthEast');     end;
        ii = 0;
        for LR = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
            filename = ['stat_' num2str(iscenario) '_' num2str(irun) '_' num2str(floor(LR*1000)) '.txt'];
            M = dlmread(filename);  % epoch_index   total_time  train_loss  val_loss    val_error
            nepochs = size(M(:,1));
            
            ii = ii + 1;
            xvals = 1:nepochs;  % epochs
            yvals = M(:,columnidx);    
            if (plotY == 3) yvals = 100.0 - yvals;  end;
            plot(xvals, yvals, 'color', colors{ii}); hold on;
            if (irun == 1)  leg{ii} = ['LR = ' num2str(LR)];    end;
        end;
    end;
    xlabel('Epochs','fontsize',16);
    ylabel(ylabeltext,'fontsize',16); 
end;


if (iscenario == 2)
    plotX = 1; % 1 - epochs, 2 - time 
    plotY = 1; % 1 - plot train_loss, 2 - plot valid_loss, 3 - plot valid_err
    if (plotX == 1)    xlabeltext = 'Epochs';   end;
    if (plotX == 2)    xlabeltext = 'Time (sec.)';   end;
    if (plotY == 1)    columnidx = 3;  ylabeltext = 'Training loss';   end;
    if (plotY == 2)    columnidx = 4;  ylabeltext = 'Validation loss';   end;
    if (plotY == 3)    columnidx = 5;  ylabeltext = 'Validation error (%)';   end;
    irun = 1;
    ii = 0;
    leg = {};
    for batch_size_train = [20, 100, 500]
        for LR = [0.1, 0.2, 0.5]
            filename = ['stat_' num2str(iscenario) '_' num2str(irun) '_' ...
                num2str(floor(LR*1000)) '_' num2str(batch_size_train) '.txt'];
            M = dlmread(filename);  % epoch_index   total_time  train_loss  val_loss    val_error
            nepochs = size(M(:,1));
            
            ii = ii + 1;
            if (plotX == 1)            xvals = 1:nepochs;   end;
            if (plotX == 2)            xvals = M(:,2);   end;
            yvals = M(:,columnidx);    
            if (plotY == 3) yvals = 100.0 - yvals;  end;
            loglog(xvals, yvals, 'color', colors{ii}); hold on;
            if (irun == 1)  leg{ii} = ['LR = ' num2str(LR) ', BS = ' num2str(batch_size_train)];    end;
        end;
    end;
    legend(leg,'Location','SouthWest');
    xlabel(xlabeltext,'fontsize',16);
    ylabel(ylabeltext,'fontsize',16); 
end;

if (iscenario == 3)
    plotX = 1; % 1 - epochs, 2 - time 
    plotY = 1; % 1 - plot train_loss, 2 - plot valid_loss, 3 - plot valid_err
    if (plotX == 1)    xlabeltext = 'Epochs';   end;
    if (plotX == 2)    xlabeltext = 'Time (sec.)';   end;
    if (plotY == 1)    columnidx = 3;  ylabeltext = 'Training loss';   end;
    if (plotY == 2)    columnidx = 4;  ylabeltext = 'Validation loss';   end;
    if (plotY == 3)    columnidx = 5;  ylabeltext = 'Validation error (%)';   end;
    irun = 1;
    ii = 0;
    leg = {};
    for Mom = [0, 0.5, 0.9]
        for LR = [0.001, 0.01, 0.1]
            filename = ['stat_' num2str(iscenario) '_' num2str(irun) '_' ...
                num2str(floor(LR*1000)) '_' num2str(floor(Mom*1000)) '.txt'];
            M = dlmread(filename);  % epoch_index   total_time  train_loss  val_loss    val_error
            nepochs = size(M(:,1));
            
            ii = ii + 1;
            if (plotX == 1)            xvals = 1:nepochs;   end;
            if (plotX == 2)            xvals = M(:,2);   end;
            yvals = M(:,columnidx);    
            if (plotY == 3) yvals = 100.0 - yvals;  end;
            loglog(xvals, yvals, 'color', colors{ii}); hold on;
            %semilogx(xvals, yvals, 'color', colors{ii}); hold on;
            if (irun == 1)  leg{ii} = ['LR = ' num2str(LR) ', M = ' num2str(Mom)];    end;
        end;
    end;
    legend(leg,'Location','SouthWest');
    xlabel(xlabeltext,'fontsize',16);
    ylabel(ylabeltext,'fontsize',16); 
end;

if (iscenario == 4)
    plotX = 1; % 1 - epochs, 2 - time 
    plotY = 1; % 1 - plot train_loss, 2 - plot valid_loss, 3 - plot valid_err
    if (plotX == 1)    xlabeltext = 'Epochs';   end;
    if (plotX == 2)    xlabeltext = 'Time (sec.)';   end;
    if (plotY == 1)    columnidx = 3;  ylabeltext = 'Training loss';   end;
    if (plotY == 2)    columnidx = 4;  ylabeltext = 'Validation loss';   end;
    if (plotY == 3)    columnidx = 5;  ylabeltext = 'Validation error (%)';   end;
    irun = 1;
    ii = 0;
    leg = {};
    for nfilters = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            filename = ['stat_' num2str(iscenario) '_' num2str(irun) '_' ...
                num2str(nfilters) '.txt'];
            M = dlmread(filename);  % epoch_index   total_time  train_loss  val_loss    val_error
            nepochs = size(M(:,1));
            
            ii = ii + 1;
            if (plotX == 1)            xvals = 1:nepochs;   end;
            if (plotX == 2)            xvals = M(:,2);   end;
            yvals = M(:,columnidx);    
            if (plotY == 3) yvals = 100.0 - yvals;  end;
            loglog(xvals, yvals, 'color', colors{ii}); hold on;
            %semilogx(xvals, yvals, 'color', colors{ii}); hold on;
            if (irun == 1)  leg{ii} = ['nfilters = ' num2str(nfilters)];    end;
    end;
    legend(leg,'Location','SouthWest');
    xlabel(xlabeltext,'fontsize',16);
    ylabel(ylabeltext,'fontsize',16); 
end;

if (iscenario == 5)
    irun = 1;
    filename = ['solutions_' num2str(iscenario) '_' num2str(irun) '.txt'];
    M = dlmread(filename);  % epoch_index   total_time  train_loss  val_loss    val_error
    
    nevaluations = size(M,1);   % number of evaluated solutions
    nsimulatedruns = 1000;        % number of simulations of random search
    for iter=1:nsimulatedruns 
        indexes = 1:nevaluations;
        simidx = randsample(indexes,nevaluations);   % random sequence without repetations
        fbest(iter,1) = 100.0 - M(simidx(1),5);
        for t=2:nevaluations    % find the best error so far at step t
            newval = 100.0 - M(simidx(t),5);
            fbest(iter,t) = fbest(iter,t-1);
            if (newval < fbest(iter,t-1))
                fbest(iter,t) = newval;
            end;
        end;
    end;
    median_fbest = median(fbest);
    semilogx(1:nevaluations, fbest(1,:), 'color', 'blue');    hold on;
    semilogx(1:nevaluations, median_fbest, 'color', 'red', 'LineWidth', 5);    hold on;
    legend('Simulated run','Median simulated run');
    nsimulationsToPlot = 20;
    for iter=2:nsimulationsToPlot
       	semilogx(1:nevaluations, fbest(iter,:), 'color', 'blue');    hold on;
    end;
    
    xlabel('Evaluations','fontsize',16);
    ylabel('Best Validation error (%)','fontsize',16);
    title('Random search','fontsize',16);
    ylim([0.5 1.5]);
end;
