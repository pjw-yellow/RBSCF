%   This script runs the implementation of EMCF, which is borrowed from
%   BACF.

clear; 
clc;
close all;
setup_paths;

%   Load video information
% base_path  = 'F:/tracking/datasets/UAVTrack112';
clear all;
base_path = 'D:/期刊/DCF/Benchmark/DTB70';
video      = {'ChasingDrones'};
% video      = {'Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark', 'CarScale', ...
%     'Coke', 'Couple', 'Crossing', 'David2', 'David3', 'David', 'Deer', ...
%     'Dog1', 'Doll', 'Dudek', 'Faceocc1', 'Faceocc2', 'Fish', 'Fleetface', ...
%     'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', ...
%     'Ironman', 'Jogging_1', 'Jogging_2','Jumping', 'Lemming', 'Liquor', 'Matrix', ...
%     'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer1', ...
%     'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv', 'Sylvester', ...
%     'Tiger1', 'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman'};
%base_path = './seq';
%video      = 'Yacht2';

for vid = 1:numel(video)
    close all;

    video_path = [base_path '/' video{vid}];
    [seq, ground_truth] = load_video_info(video,base_path,video_path);
    seq.path = video_path;
    seq.name = video;
    seq.startFrame = 1;
    seq.endFrame = seq.len;

    gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)]; 

    %   Run EMCF
    results       = run_Oursh(seq);

    %   compute the OP
    pd_boxes = results.res;
    pd_boxes = [pd_boxes(:,1:2), pd_boxes(:,1:2) + pd_boxes(:,3:4) - ones(size(pd_boxes,1), 2)  ];
    
    OP = zeros(size(gt_boxes,1),1);
    for i=1:size(gt_boxes,1)
        b_gt = gt_boxes(i,:);
        b_pd = pd_boxes(i,:);
        OP(i) = computePascalScore(b_gt,b_pd);
    end
    OP_vid = sum(OP >= 0.5) / numel(OP);
    
    FPS_vid = results.fps;
    display([video  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(OP_vid)]);
    
    if vid==1
        gt_boxes_t=gt_boxes; 
        pd_boxes_t=pd_boxes;
    end
    gt_boxes_t=cat(1, gt_boxes_t, gt_boxes); 
    pd_boxes_t=cat(1, pd_boxes_t, pd_boxes);
end    

OP = zeros(size(gt_boxes_t,1),1);
for i=1:size(gt_boxes_t,1)
    b_gt = gt_boxes_t(i,:);
    b_pd = pd_boxes_t(i,:);
    OP(i) = computePascalScore(b_gt,b_pd);
end
OP_vid = sum(OP >= 0.5) / numel(OP);
%FPS_vid = results.fps;
%display([video  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(OP_vid)]);
precisions = precision_plot(pd_boxes_t, gt_boxes_t, 'MY', 1);