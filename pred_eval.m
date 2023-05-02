s = filesep;
skin_threshold_range = 0.6:0.1:0.6;
skin_percentage_range = 0.09:0.01:0.12;
scales = .2:.2:2;
weak_classifiers = load('weak_classifiers.mat').weak_classifiers;
best_classifiers = load('best_classifiers.mat').best_classifiers;
face_list = dir(['data', s,'training_faces', s, '*.bmp']);
train_image = read_gray([face_list(1).folder ,s, face_list(1).name]);
[face_vertical,face_horizontal ] = size(train_image);

%% read annotations
pattern = ['data', s , 'test_face_photos', s, '*.jpg'];
face_photos = dir(pattern);
annotation_file_name = 'face_annotations.txt';
fid = fopen(['data', s , 'test_face_photos', s, annotation_file_name], 'r');

line = fgetl(fid);
annotations = {};
while ischar(line)
    if startsWith(line,'%')
        line = fgetl(fid);
        continue
    end
    temp_fields = strsplit(line, '=');

    fields = strsplit(temp_fields{2}(2:end), ' ');
    photo_file_name = fields{1}(3:end-2);

    num_faces = (length(fields) - 1) / 4;
    faces = {};
    for i = 1:num_faces
        face_top = str2double(fields{(i-1)*4 + 2}(2:end));
        face_bottom = str2double(fields{(i-1)*4 + 3});
        face_left = str2double(fields{(i-1)*4 + 4});
        if i == num_faces
            face_right = str2double(fields{(i-1)*4 + 5}(1:end-3));
        else
            face_right = str2double(fields{(i-1)*4 + 5}(1:end-2));
        end
        faces{i} = [face_top, face_bottom, face_left, face_right]; %#ok<AGROW> 
    end

    annotation = {photo_file_name, faces};
    annotations{end+1} = annotation;

    line = fgetl(fid);
end

% Close the txt file
fclose(fid);

%% Calculating whole aray
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');
row_counter = 1;
for skin_threshold = skin_threshold_range
    for skin_percentage = skin_percentage_range
        box_accepted = zeros(size(face_photos));
        for photo_no = 1:size(face_photos)
            face_photo_path = [face_photos(photo_no).folder ,s, face_photos(photo_no).name];
            color_face_photo = imread(face_photo_path);
            grey_face_photo = read_gray(face_photo_path);
            result_number = 10;
            boxes = cascading(best_classifiers, weak_classifiers, [face_vertical,face_horizontal], scales, face_photo_path);
            
            % Apply skin detection
            for box_no = 1:result_number
                coord = boxes(box_no, 1:4);
                if any(coord(:)<1)
                    continue
                end

                boxes(box_no, 2) = min(boxes(box_no, 2) , size(color_face_photo,1));
                boxes(box_no, 4) = min(boxes(box_no, 4) , size(color_face_photo,2));
        
                image = imresize(color_face_photo(boxes(box_no, 1): boxes(box_no, 2), ...
                                    boxes(box_no, 3): boxes(box_no, 4), :), [face_vertical,face_horizontal]);
                size(image);
                skin_detection_win = detect_skin(image, positive_histogram,  negative_histogram);
                skin_pixels = skin_detection_win > skin_threshold;
                if sum(sum(skin_pixels(skin_pixels==1))) / (size(skin_pixels,1)*size(skin_pixels,2)) > skin_percentage
                    box_accepted(photo_no) = box_accepted(photo_no) + 1;
                    grey_face_photo = draw_rectangle1(grey_face_photo, boxes(box_no, 1), boxes(box_no, 2), ...
                                            boxes(box_no, 3), boxes(box_no, 4));
                end
            end
            figure(photo_no);imshow(grey_face_photo,[]);
% 
%             dir_name = 'outputs';
%             if exist(dir_name, 'dir') == 0
%                 mkdir(dir_name);
%                 disp(['Directory ', dir_name, ' created.']);
%             else
%                 disp(['Directory ', dir_name, ' already exists.']);
%             end
% 
%             imwrite(grey_face_photo/255,['outputs',s, face_photos(photo_no).name],'png');
        end
        errors =  zeros(size(face_photos));
        for photo_no = 1:size(box_accepted)
            for ano_no = 1:size(annotations,2)
                if strcmp(annotations{ano_no}(1),face_photos(photo_no).name)
                    errors(photo_no) = abs(box_accepted(photo_no) - size(annotations{ano_no}{2},2));
                end
            end
        end
        disp([skin_threshold skin_percentage sum(errors)]);
    end
end

