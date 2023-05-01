%% cascading
% making groups out of the best classifer
function boxes = cascading(best_classifiers, face_size, scales, img_path, num_boxes) 
tic

group_1 = best_classifiers(1:7, :);
group_2 = best_classifiers(8:18, :);
group_3 = best_classifiers(19:43, :);

% taking a image window and passing it to the detectors

% need to pass the coordinates of of the window 
%face_size = [100 100];
%for all the windows

%image = read_gray('data/test_face_photos/DSC02950.JPG');
image = read_gray(img_path);
size(image)

face_vertical = face_size(1);
face_horizontal = face_size(2);

%scales = .5:.25:2;

max_responses = ones(size(image)) * -10;
max_scales = zeros(size(image));

for scale = scales
    scaled_image = imresize(image, 1/scale, 'bilinear');
    integral = integral_image(scaled_image);
    vertical_size = size(scaled_image, 1);
    horizontal_size = size(scaled_image, 2);
    
    window_coords = zeros((vertical_size-face_vertical)*(horizontal_size -face_horizontal),2);
    for i =  1:(vertical_size-face_vertical+1)
        curr_point = (i-1) * (horizontal_size - face_horizontal + 1);
        for j = 1:(horizontal_size - face_horizontal + 1)
            window_coords(curr_point + j,1) = i;
            window_coords(curr_point + j,2) = j;
        end
    end
    
    [~ , group_1_selection] = filter_window(group_1, weak_classifiers, window_coords, integral, vertical_size, horizontal_size, face_size);
    [~ , group_2_selection] = filter_window(group_2, weak_classifiers, group_1_selection, integral, vertical_size, horizontal_size, face_size);
    [responses, ~] = filter_window(group_3, weak_classifiers, group_2_selection, integral, vertical_size, horizontal_size, face_size);
    temp_result = imresize(responses, size(image), 'bilinear');
    
    higher_maxes = (temp_result > max_responses);
    max_scales(higher_maxes) = scale;
    max_responses(higher_maxes) = temp_result(higher_maxes);
end

boxes = detection_boxes(image, zeros(face_size), max_responses, ...
                         max_scales, num_boxes);

% result = image;
% 
% for number = 1:5
%     result = draw_rectangle1(result, boxes(number, 1), boxes(number, 2), ...
%                              boxes(number, 3), boxes(number, 4));

toc
% figure(1);
% imshow(result,[]);
end



   

