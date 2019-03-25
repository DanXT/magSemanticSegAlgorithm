% batching the IMEX network

dataSize     = [500,5000];
numTimeSteps = [2,4,8,16];
width        = [16,32,64];

for i1=1:2
    for i2=1:4
        for i3 = 1:4
            fprintf('data size %3d   time steps %3d  width %3d\n', ...
                     dataSize(i1),numTimeSteps(i2),width(i3))
            runManyImplicitEamples(dataSize(i1),numTimeSteps(i2),width(i3));
        end
    end
end