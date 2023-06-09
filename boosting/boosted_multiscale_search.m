function [max_responses, max_scales] = ...
    boosted_multiscale_search(image, scales, ...
                              classifiers, weak_classifiers, face_size)

% function [max_responses, max_scales] = ...
%     boosted_multiscale_search(image, scales, ...
%                               classifiers, weak_classifiers, ...
%                               face_size, result_number)
%
% for each pixel, search over the specified scales and rotations,
% and record:
% - in result, the max response score for that pixel over all scales
% - in max_scales, the scale that gave the max score

max_responses = ones(size(image)) * -10;
max_scales = zeros(size(image));

for scale = scales
%     scaler
%     tic;
    scaled_image = imresize(image, 1/scale, 'bilinear');
    temp_result = apply_classifier_aux(scaled_image, classifiers, ...
                                       weak_classifiers, face_size);
    temp_result = imresize(temp_result, size(image), 'bilinear');
    
    higher_maxes = (temp_result > max_responses);
    max_scales(higher_maxes) = scale;
    max_responses(higher_maxes) = temp_result(higher_maxes);
%     toc
end
