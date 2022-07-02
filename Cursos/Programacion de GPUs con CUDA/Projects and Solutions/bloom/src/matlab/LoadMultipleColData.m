b=[];
fid = fopen('unif_stats.cpu', 'r');
while 1
    tline=fgetl(fid);
    if ~ischar(tline), break, end
    a=str2num(tline);
    b=[b;a];
end;
fclose(fid);

figure(1);
plot(b(:,1),b(:,3));
hold on;
plot(b(:,1),b(:,4),'r');
xlabel('ID of random sequences');
ylabel('Occurrences of random sequences');
title('Estimated occurrences of PBF GPU');


figure(2);
plot(b(:,1),b(:,5));
xlabel('ID of random sequences');
ylabel('Relative error');
title('Accuracy of PBF GPU');
