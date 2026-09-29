function UAV_R1_Figures(outDir)
% UAV_R1_FIGURES  Reproduce R1 evidence figures in MATLAB (R2020b or newer).
% Place this file and UAV_R1_Figure_Data.mat in the same folder, then run:
%   UAV_R1_Figures
% Optional output directory: UAV_R1_Figures('C:\my_figures')
% No simulation, fitting or synthetic evidence is performed here.
% Outputs: editable MATLAB .fig, vector .pdf, and .svg for five figures.
% All numeric data and source hashes are in the accompanying MAT file.
% Layout revision 2: based on the five user-supplied MATLAB SVG exports.
% MATLAB is unavailable in the editing environment; rerun this revision in
% MATLAB and inspect the exports at final journal size before submission.
%
% Evidence scopes are kept separate:
%  1. Paired 100-trial frame BER, fixed/randomized single jammers.
%  2. Pulse conditional BER AND retained fraction.
%  3. Exp9b, 640-stream BLER versus counted ALE work.
%  4. Frozen v5 validation, 1200 streams, selected refresh versus periodic.
%  5. Archived v3 KV260 front-end kernel energy, five independent repeats.
% The ongoing matched hardware run is not present in these data.
% No joint waveform/spike/gate trace is invented from aggregate statistics.
%
% MATLAB export references:
% https://www.mathworks.com/help/matlab/ref/exportgraphics.html
% https://www.mathworks.com/help/matlab/ref/print.html

root = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outDir)
    outDir = fullfile(root,'R1_MATLAB_Figures');
end
if ~exist(outDir,'dir'), mkdir(outDir); end
dataPath = fullfile(root,'UAV_R1_Figure_Data.mat');
assert(exist(dataPath,'file') == 2, 'Place UAV_R1_Figure_Data.mat beside this M file.');
S = load(dataPath,'D'); D = S.D;
assert(strcmp(D.schema,'UAV_R1_MATLAB_FIGURES_20260928_v1'),'Unexpected data schema.');
assert(isequal(size(D.ber.mean),[2 4 5 11]),'Unexpected BER array shape.');
assert(isequal(size(D.energy.repeats),[4 3 5]),'Unexpected hardware repeat array.');
assert(D.validation.n_streams == 1200 && D.stream.n_streams == 640,'Panel mismatch.');

% Consistent colors. Marker/line differences also support grayscale reading.
C = [0.46 0.46 0.46; 0.80 0.46 0.10; 0.00 0.45 0.70; ...
     0.00 0.56 0.38; 0.65 0.18 0.42];
zeroY = 2e-7;             % display lane only, never inserted into the data
x = D.ber.jsr(:);

% 1. Structured and broadband interference. Pulse has its own figure.
f = newFig('Single-jammer BER',17.8,13.0);
jammers = [1 2 4]; handles = gobjects(1,5); marker = {'s','d','^','v','o'};
lineStyles = {'-','--','-.',':','-'};
for c = 1:2
    for q = 1:3
        ax = axesOf(f,[0.08+(q-1)*0.307, 0.635-(c-1)*0.385, 0.247, 0.265]);
        hold(ax,'on');
        for m = 1:5
            y = reshape(D.ber.mean(c,jammers(q),m,:),[],1);
            ci = reshape(D.ber.ci95_half(c,jammers(q),m,:),[],1);
            h = berLine(ax,x,y,ci,C(m,:),marker{m},zeroY,ismember(m,[3 5]));
            set(h,'LineStyle',lineStyles{m});
            if c == 1 && q == 1, handles(m) = h; end
        end
        berAxis(ax,zeroY); xlim(ax,[0 20]); xticks(ax,0:4:20);
        title(ax,sprintf('(%c) %s, %s','a'+(c-1)*3+q-1,...
            D.ber.types{jammers(q)},shortCondition(c)), 'FontWeight','normal');
        if c==2, xlabel(ax,'Nominal JSR (dB)'); end
        if q==1, ylabel(ax,'BER'); end
    end
end
legendLabels = {'No correction','Notch (fixed only)','ALE 128 (\mu = 0.02)',...
    'LSTM (0-20 dB)','Proposed frame MoE'};
lg = legend(handles,legendLabels,'NumColumns',3,'Box','off','FontSize',7.7,...
    'FontName','Times New Roman','Interpreter','tex');
lg.Units='normalized'; lg.Position=[0.08 0.106 0.89 0.067];
foot(f,['100 trials per point. Whiskers: archived nominal 95% intervals for ALE and MoE.' newline ...
    'Downward triangles in the zero lane denote no observed errors. Intervals crossing zero stop at the display limit.']);
saveAll(f,outDir,'ber_single_jammers',D);

% 2. Pulse results use the same retained-bit denominator in both top panels.
f = newFig('Pulse recovery and retention',17.8,10.5);
for c = 1:2
    ax = axesOf(f,[0.11+(c-1)*0.47 0.59 0.35 0.28]); hold(ax,'on');
    y=reshape(D.ber.mean(c,3,5,:),[],1);
    ci=reshape(D.ber.ci95_half(c,3,5,:),[],1);
    berLine(ax,x,y,ci,C(5,:),'o',zeroY,true); berAxis(ax,zeroY);
    xlim(ax,[0 20]); xticks(ax,0:4:20);
    title(ax,sprintf('(%c) %s','a'+c-1,D.conditions{c}),'FontWeight','normal');
    ylabel(ax,'BER on retained bits');
    ax = axesOf(f,[0.11+(c-1)*0.47 0.20 0.35 0.25]); hold(ax,'on');
    plot(ax,x,100*D.ber.pulse_retention(c,:),'-o','Color',C(5,:),...
        'LineWidth',1.2,'MarkerSize',4,'MarkerFaceColor','w');
    xlim(ax,[0 20]); xticks(ax,0:4:20); ylim(ax,[0 105]); yticks(ax,0:25:100);
    xlabel(ax,'Nominal JSR (dB)'); ylabel(ax,'Retained bits (%)');
    title(ax,sprintf('(%c) Retention','c'+c-1),'FontWeight','normal');
end
foot(f,['Proposed frame MoE, 100 trials per point. BER is conditional on surviving bits.' newline ...
    'No observed bit errors does not imply delivery of all transmitted data. Retention intervals were not archived.']);
saveAll(f,outDir,'pulse_ber_and_retention',D);

% 3. All eight receivers from one 640-stream panel, without mixing v5 data.
f = newFig('Streaming quality and ALE work',17.8,10.0);
SC=[C(5,:);C(5,:);0.45 0.16 0.50; C(3,:);C(3,:); C(2,:);C(4,:);C(1,:)];
marks={'o','s','d','o','s','^','v','x'}; handles=gobjects(1,8);
% Concentric open symbols expose nearly coincident periodic/refresh results.
% Only marker sizes differ; the measured x and y coordinates are not moved.
markerSizes=[7.6 4.2 5.6 7.6 4.2 5.8 5.8 6.0];
dy=[1.2 -1.4 1.1 1.5 -1.3 1.5 -1.3 0.6];
for c=1:2
    ax=axesOf(f,[0.10+(c-1)*0.48 0.29 0.36 0.58]); hold(ax,'on');
    for m=1:8
        xx=D.stream.values(c,m,2); yy=D.stream.values(c,m,1);
        h=plot(ax,xx,yy,marks{m},'Color',SC(m,:),'LineWidth',1.2,...
            'MarkerSize',markerSizes(m),'MarkerFaceColor','none');
        if c==1,handles(m)=h;end
        text(ax,xx+8,yy+dy(m),num2str(m),'FontName','Times New Roman',...
            'FontSize',8,'Color',SC(m,:),'VerticalAlignment','middle');
    end
    xlim(ax,[-15 420]); ylim(ax,[0 41]); xticks(ax,0:100:400);
    xlabel(ax,'ALE tap products / input sample'); ylabel(ax,'BLER (%)');
    title(ax,sprintf('(%c) %s','a'+c-1,D.conditions{c}),'FontWeight','normal');
end
labels=cell(1,8);for m=1:8,labels{m}=sprintf('%d  %s',m,D.stream.labels{m});end
lg=legend(handles,labels,'NumColumns',4,'Box','off','Interpreter','none',...
    'FontName','Times New Roman','FontSize',7.6);
lg.Units='normalized';lg.Position=[0.04 0.12 0.94 0.085];
foot(f,'640 streams. Points show pooled BLER and counted ALE work, not total computation or measured energy.');
saveAll(f,outDir,'stream_bler_vs_ale_work',D);

% 4. Five fixed comparisons in a distinct, frozen 1200-stream validation.
f=newFig('Frozen refresh validation',17.8,8.5);
VC=[C(5,:);C(5,:)*0.7+0.3;C(3,:);C(3,:)*0.7+0.3;C(1,:)];
positions=[0.19 0.29 0.23 0.57;0.47 0.29 0.22 0.57;0.76 0.29 0.20 0.57];
for k=1:3
    ax=axesOf(f,positions(k,:)); hold(ax,'on');
    if k<=2, vals=reshape(D.validation.values(k,:,1),[],1);
    else, vals=reshape(sum(D.validation.values(:,:,3),1),[],1); end
    b=barh(ax,1:5,vals,0.67,'FaceColor','flat','EdgeColor','none');b.CData=VC;
    set(ax,'YDir','reverse','YTick',1:5,'YLim',[0.35 5.65]);
    if k==1,set(ax,'YTickLabel',D.validation.labels);else,set(ax,'YTickLabel',{});end
    if k<=2
        xlim(ax,[0 10.7]);xticks(ax,[0 5 10]);xlabel(ax,'BLER (%)');
        title(ax,sprintf('(%c) %s','a'+k-1,shortCondition(k)),'FontWeight','normal');
        for m=1:5,text(ax,vals(m)+0.13,m,sprintf('%.2f',vals(m)),...
                'FontSize',7.6,'FontName','Times New Roman','VerticalAlignment','middle');end
    else
        xlim(ax,[0 51000]);xticks(ax,[0 20000 40000]);
        set(ax,'XTickLabel',{'0','20k','40k'});xlabel(ax,'Calls, both regimes');
        title(ax,'(c) Classifier calls','FontWeight','normal');
        for m=1:5
            if vals(m)==0
                label='0';
            else
                label=sprintf('%.1fk',vals(m)/1000);
            end
            text(ax,vals(m)+900,m,label,'FontSize',7.2,...
                'FontName','Times New Roman','VerticalAlignment','middle');
        end
    end
end
saving=100*(1-sum(D.validation.values(:,2,3))/sum(D.validation.values(:,1,3)));
foot(f,sprintf(['Frozen validation: 1200 streams. Spike refresh reduces classifier calls by %.2f%%.' newline ...
    'Classifier-call counts exclude other computation. Analog-delta and fast-rule controls remain visible.'],saving));
saveAll(f,outDir,'frozen_refresh_validation',D);

% 5. Five-repeat kernel measurements and explicitly paired contrasts.
f=newFig('KV260 front-end kernel energy',17.8,10.0);
EC=[0.76 0.43 0.12;0.00 0.42 0.65;0.53 0.29 0.61];
ax=axesOf(f,[0.09 0.33 0.43 0.53]);hold(ax,'on');
B=bar(ax,D.energy.mean,'grouped','BarWidth',0.83,'EdgeColor','none');
for m=1:3,B(m).FaceColor=EC(m,:);end
drawnow;
for m=1:3
    centers=B(m).XEndPoints(:); yy=D.energy.mean(:,m);
    lo=yy-reshape(D.energy.ci95(:,m,1),[],1);
    hi=reshape(D.energy.ci95(:,m,2),[],1)-yy;
    errorbar(ax,centers,yy,lo,hi,'k.','LineWidth',0.75,'CapSize',4,'HandleVisibility','off');
    for v=1:4
        jitter=linspace(-0.035,0.035,5);
        plot(ax,centers(v)+jitter,reshape(D.energy.repeats(v,m,:),1,[]),'.',...
            'Color',[0.25 0.25 0.25],'MarkerSize',5,'HandleVisibility','off');
    end
end
set(ax,'XTick',1:4,'XTickLabel',D.energy.labels,'XTickLabelRotation',0);ylim(ax,[0 10.5]);
ylabel(ax,'nJ / input sample / replica');title(ax,'(a) Attributed kernel energy','FontWeight','normal');
lg=legend(B,D.energy.kernel_labels,'Box','off','NumColumns',1,...
    'FontName','Times New Roman','FontSize',7.8,'Interpreter','none');
lg.Units='normalized';lg.Position=[0.12 0.105 0.37 0.125];
ax=axesOf(f,[0.65 0.33 0.32 0.53]);hold(ax,'on');hh=gobjects(1,2);
for m=1:2
    centers=(1:4)'+(m-1.5)*0.16;yy=D.energy.paired_mean(:,m);
    lo=yy-reshape(D.energy.paired_ci95(:,m,1),[],1);
    hi=reshape(D.energy.paired_ci95(:,m,2),[],1)-yy;
    color=EC(3,:);if m==2,color=EC(1,:);end
    hh(m)=errorbar(ax,centers,yy,lo,hi,'o','Color',color,'LineWidth',1.0,...
        'MarkerSize',4,'MarkerFaceColor','w','CapSize',4);
end
yline(ax,0,':','Color',[0.3 0.3 0.3],'HandleVisibility','off');
set(ax,'XTick',1:4,'XTickLabel',D.energy.labels,'XTickLabelRotation',0);
xlim(ax,[0.5 4.5]);ylim(ax,[-0.3 9]);
ylabel(ax,'Paired difference (nJ/sample/replica)');
title(ax,'(b) Difference from gated','FontWeight','normal');
lg=legend(hh,{'FFT - gated','Dense - gated'},'Box','off','FontName','Times New Roman','FontSize',7.8);
lg.Units='normalized';lg.Position=[0.66 0.12 0.30 0.09];
foot(f,['Archived KV260 v3, approximately 1 MS/s. Five repeats, idle and D0 adjustment, nominal 95% t intervals.' newline ...
    'Front-end kernels only. Clean and tone are fixed; pulse and switching are randomized.']);
saveAll(f,outDir,'kv260_frontend_kernel_energy',D);

fprintf('Tamamlandi. MATLAB sekilleri: %s\n',outDir);
fprintf('Her sekil icin .fig, vektor .pdf ve .svg olusturuldu.\n');
fprintf('Zaman izli mekanizma sekli uretilmedi: ortak ham kayit mevcut degil.\n');
end

function f=newFig(name,w,h)
f=figure('Name',name,'NumberTitle','off','Color','w','Units','centimeters',...
    'Position',[2 2 w h],'PaperUnits','centimeters','PaperSize',[w h],...
    'PaperPosition',[0 0 w h]);
end
function ax=axesOf(f,p)
ax=axes('Parent',f,'Units','normalized','Position',p,'FontName','Times New Roman',...
    'FontSize',8.3,'LineWidth',0.65,'Box','on','TickDir','out',...
    'XColor',[0.1 0.1 0.1],'YColor',[0.1 0.1 0.1],'GridColor',[0.75 0.75 0.75],...
    'GridAlpha',0.30,'GridLineStyle','-','Layer','top');grid(ax,'on');
set(ax,'XMinorGrid','off','YMinorGrid','off');
end
function s=shortCondition(k)
if k==1,s='fixed';else,s='randomized';end
end
function h=berLine(ax,x,y,ci,col,mk,zeroY,showCI)
lineY=y;lineY(y==0)=NaN;
h=plot(ax,x,lineY,['-' mk],'Color',col,'LineWidth',1.1,'MarkerSize',3.8,'MarkerFaceColor','w');
if showCI
    ii=isfinite(y)&y>0&isfinite(ci);
    low=min(ci(ii),max(y(ii)-zeroY,0));
    errorbar(ax,x(ii),y(ii),low,ci(ii),'.','Color',col,'LineWidth',0.65,...
        'CapSize',3,'HandleVisibility','off');
end
ii=isfinite(y)&y==0;
if any(ii)
    plot(ax,x(ii),zeroY*ones(sum(ii),1),'v','Color',col,'MarkerSize',4.2,...
        'MarkerFaceColor','w','LineStyle','none','HandleVisibility','off');
end
end
function berAxis(ax,zeroY)
set(ax,'YScale','log','YMinorGrid','off','XMinorGrid','off');ylim(ax,[zeroY*0.65 0.7]);
yticks(ax,[zeroY 1e-6 1e-4 1e-2]);
yticklabels(ax,{'0 obs.','10^{-6}','10^{-4}','10^{-2}'});
end
function foot(f,msg)
annotation(f,'textbox',[0.03 0.001 0.94 0.075],'String',msg,'EdgeColor','none','Margin',0,...
    'FontName','Times New Roman','FontSize',7.0,'Interpreter','none',...
    'HorizontalAlignment','center','VerticalAlignment','middle','FitBoxToText','off');
end
function saveAll(f,outDir,name,D)
set(f,'UserData',struct('source_schema',D.schema,'figure_name',name,...
    'layout_revision',2,'source_provenance',{D.provenance}));
drawnow;
savefig(f,fullfile(outDir,[name '.fig']));
exportgraphics(f,fullfile(outDir,[name '.pdf']),'ContentType','vector','BackgroundColor','white');
print(f,fullfile(outDir,[name '.svg']),'-dsvg');
end
