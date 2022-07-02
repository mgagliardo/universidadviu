%% Test 1
M = csvread('rand.txt');
figure(1);
hist(M,1000);
xlabel('ID of random sequences');
ylabel('Occurrences of random sequences');
title('Random distribution of sequences');

%% Test 2
M = csvread('unif.txt');
figure(2);
hist(M);
xlabel('ID of random sequences');
ylabel('Occurrences of random sequences');
title('Uniform distribution of sequences');

%% Test 3
M = csvread('norm.txt');
figure(3);
hist(M,100);
xlabel('ID of random sequences');
ylabel('Occurrences of random sequences');
title('Normal distribution of sequences');

%% Test 4
M = csvread('pois.txt');
figure(4);
hist(M,1000);
xlabel('ID of random sequences');
ylabel('Occurrences of random sequences');
title('Poisson distribution of sequences');
