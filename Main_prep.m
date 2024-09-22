modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
  "16QAM", "32QAM", "64QAM", "GFSK", ...
  "B-FM", "DSB-AM", "SSB-AM"]);
sps = 8;                % Samples per symbol
spf = 1024;             % Samples per frame
fs = 200e3;             % Sample rate
numModulationTypes = length(modulationTypes);

init_seed{1,1} = zeros(256:1,"double");


 for modType = 1:numModulationTypes
     dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
         
     % Generate random data
     init_seed{1,modType} = [dataSrc()];
 
 end
 