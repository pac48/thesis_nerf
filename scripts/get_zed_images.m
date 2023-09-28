clear all;

zed = ZED_Camera();
mexZED('setCameraSettings', 'exposure', 60, 0)
mexZED('setCameraSettings', 'gain', 0, 0)
mexZED('setCameraSettings', 'brightness', 5, 0)
mexZED('setCameraSettings', 'contrast', 2, 0)

all_imgs = {};
ind = 1;
tic

angular_velocity = zeros(1, 3);
alpha = 0.1;
while 1
    angular_velocity = (1-alpha)*angular_velocity  + (alpha)*mexZED('getSensorsData', 1).IMUData.angular_velocity;
    val = norm(angular_velocity)
    if val < .5 && toc > .5
        tic
        [image_left, image_right] = zed.read_stereo();
        subplot(2,2,1)
        imshow(image_left);
        title('Image Left')
        subplot(2,2,2)
        imshow(image_right);
        title('Image Right')
        subplot(2,2,3)
        drawnow
        all_imgs{ind} = image_left;
        ind = ind+1;
        all_imgs{ind} = image_right;
        ind = ind+1;
    end

end


for i = 1:length(all_imgs)
    img = all_imgs{i};
    imwrite(img, ['imgs/' num2str(i+78) '.png'])
end