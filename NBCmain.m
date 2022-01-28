%% Main script to implement Naive Bayes Classifer to classify text 
% messages as SPAM or not SPAM
clear all
close all
%% Let's start with importing the training data
T=readtable('NBC_training_data.csv');
Tstr=T.text; %import table as vector of n strings
Tspam=T.spam;

%There are 4000 samples in the trianing data with the first column
%representing the label (True of False spam) and the second column
%consisting of the text string

for i=1:length(Tspam)
   if Tspam{i}=="FALSE"
    y(i,1)=0;
   end
   if Tspam{i}~="FALSE"
   y(i,1)=1;
   end
end
clear Tspam T 

%Next, to reduce the number of features and subsequently the computation time,
%I dabble in some training data preprocessing by: 
%(1) removing words/letters/numbers that aren’t meaningful indicators
%of whether a message is spam or not (like: ‘a’,’e’,’i’, and all numbers) , 
%(2) changing the entire training dataset into lowercase, and (3) removing 
%special characters (like: !@#$%^) as well as replacing periods (.) with spaces ( ).

Tstr=lower(Tstr); %lower case all
n=length(y);
for i=1:n
    Tstrtemp=Tstr(i,1);
    Tstrnew(i,:)=erase(Tstrtemp,'!');
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'@');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'#');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'$');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'%');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'^');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'&');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'*');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'(');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,')');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'-');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'_');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'+');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'=');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'`');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'~');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'<');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,',');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'.');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'>');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'/');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'?');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'''');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'"');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,':');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,';');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'\');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'1');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'2');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'3');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'4');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'5');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'6');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'7');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'8');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'9');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'0');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'a');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'e');   
    Tstrtemp=Tstrnew(i,:);
    Tstrnew(i,:)=erase(Tstrtemp,'i');   
    Tstrtemp=Tstrnew(i,:);
end

%The dimension reduction consequences of the above-mentioned training
%dataset preprocessing framework are twofold. First, by removing a 
%subset of possible characters in the text message, the size our of 
%training data word dictionary is significantly reduced. To quantify 
%with numbers, we go from 10964 unique words in our training dictionary
%to around 6053 after removing 42 characters from the training data. 
%Secondly, removing common vowels like ‘a’, ‘e’, and ‘i’ helps reduce
%the number of permutations of unique words that could be generated 
%from an accidental typo in a text message. Listed below are some 
%examples of text messages before and after preprocessing:

Tnew="";
Treadmirrorlabel=0;
for i=1:n
    Tread=split(Tstrnew(i,:))';
    Treadmirrorlabel=[Treadmirrorlabel, y(i)*ones(length(Tread),1)'];
    Tnew=[Tnew,Tread];  
end

%% Example 1 (spam, before)
%'free entry into our å£250 weekly comp just send the word win to 80086 now. 18 t&c www.txttowin.co.uk'
%% Example 1 (spam, after)
%'fr ntry nto our å£ wkly comp just snd th word wn to now tc wwwtxttowncouk'

%% Example 2 (not spam, before)
%'but i juz remembered i gotta bathe my dog today..'
%% Example 2 (not spam, after)
%'but juz rmmbrd gott bth my dog tody'

%% To generate our spam dictionary using the processed training data,
%I identify all unique words in the training data as well as their
%corresponding frequency. To achieve this each text message string
%is divided into an array of characters and their attributes are
%parsed into the dictionary using functions like: unique() and accumarray().

%% calculate the frequency of words used in all of training data
[ii,jj,kk]=unique(Tnew);
freq=accumarray(kk,1);
ii=ii';
[freq,sortIdx] = sort(freq,'descend');
ii = ii(sortIdx);

spamTnew=Tnew(Treadmirrorlabel==1);
[iispam jjspam kkspam]=unique(spamTnew);
freqspam=accumarray(kkspam,1);
iispam=iispam';
[freqs,sortIdx]=sort(freqspam,'descend');
iispam=iispam(sortIdx);

nspamTnew=Tnew(Treadmirrorlabel==0);
[iinspam jjnspam kknspam]=unique(nspamTnew);
freqnspam=accumarray(kknspam,1);
iinspam=iinspam';
[freqns,sortIdx]=sort(freqnspam,'descend');
iinspam=iinspam(sortIdx);

%% This process is repeated twice separately for the subset of 
%spam and not spam messages. This allows us to create two 
%dictionaries: a frequency-based descending list of spam words
%and not spam words.

%% Training Model

py=sum(y)/length(y); %probablity that a email is spam as estimated from the training data

% Let's find the probability of words in a spam message are words in our
% dictionary

px4yspam=(freqspam+1)/(sum(freqspam)+2);
px4ynspam=(freqnspam+1)/(sum(freqnspam)+2);

%% TEST
teststr="m ww gy hm";
teststr=split(teststr);
px4ys=1;
px4yns=1;
for i=1:length(teststr)
    qq=find(teststr(i)==iispam);
    if isempty(qq)
    qq=length(iispam);
    
    end
    px4ys=px4ys+px4yspam(qq);
    pp=find(teststr(i)==iinspam);
    if isempty(pp)
    pp=length(iinspam);
    end
    px4yns=px4yns+px4ynspam(pp);
end
spamprob=(px4ys)*(py);
notspamprob=(px4yns)*(1-py);
