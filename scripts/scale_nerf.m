fileName = '/home/paul/CLionProjects/thesis_nerf/nerf/transforms.json';
str = fileread(fileName); % dedicated for reading files as text 
data = jsondecode(str);
% scale = 1/3.802469136;
scale = 1/3;

frames = data.frames;
for i = 1:length(frames )
frames(i).transform_matrix(1:3,end) = frames(i).transform_matrix(1:3,end)*scale;

end

data.frames = frames;

str = jsonencode(data,PrettyPrint=true)
