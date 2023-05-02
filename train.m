
%%

%%making training face images
s = filesep;
crop_ratio = 1;
face_path = ['data' ,s,'training_faces'];
train_face_file = 'training_face';
make_face_mat(face_path, train_face_file, crop_ratio);

%making training nonface images
nonface_path = ['data' ,s,'training_nonfaces'];
train_nonface_file = 'training_nonface';
cropping_image(crop_ratio, nonface_path, face_path, train_nonface_file);

face = load('training_face.mat').face_images;
non_face = load('training_nonface.mat').nonface_images;

face_vertical = size(face,1);
face_horizontal = size(face, 2);

face_size = [face_vertical, face_horizontal];

%generating weak classifiers
number = 2000;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end


% integral image
training_data = cat(3, face, non_face);
total_train_size = size(training_data,3); 
integral_train = zeros(size(training_data));

%generating labels
labels = zeros(total_train_size,1);
for i = 1 : total_train_size
    if i < size(face,3)+1
        labels(i) = 1;
    else
        labels(i) = -1;
    end
end

%permutating labels
perm = randperm(total_train_size);
training_data = training_data(: , : , perm);
labels = labels(perm , :);

%spliting data and labels

training_set_size = round(total_train_size * .7);
train_set = training_data(:, :, 1:training_set_size);
train_labels = labels(1:training_set_size, :);

validation_set_size = total_train_size - training_set_size;
validation_labels = labels(training_set_size+1:end , :);
valid_set = training_data(:, : , training_set_size+1:end);
best_classifiers = [];
session  = 5;
for i = 1: session
%getting preliminary responses from the weak classifiers
responses =  zeros(number, training_set_size);
for example = 1:training_set_size
    integral = integral_image(train_set(:,:,example));
    for feature = 1:number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
end

% adaboost
rounds = 43;
best_classifiers = AdaBoost(responses, train_labels, rounds);


% predicting of validation set
temp_responses = zeros(validation_set_size,1);
final_responses = ones(validation_set_size, 1) * -1;

for i = 1:size(valid_set,3)
    image = valid_set(: , :, i);
    [max_response , ~] = boosted_multiscale_search(image, 1, best_classifiers, weak_classifiers, [face_vertical, face_horizontal]);
    temp_responses(i) = sum(max_response(:));
end

%finding mismatch validation indexes
face_response = temp_responses > 0;
final_responses(face_response) = 1;
mismatch = final_responses ~= validation_labels;
mismatch_ind = find(mismatch==1);

%ending if no wrong prediction
if isempty(mismatch_ind)
    break
end

%seperating mismatched data
mismatch_data = valid_set(:, :, mismatch_ind);
mismatched_labels = validation_labels(mismatch_ind, :);
train_set  = cat(3,train_set,mismatch_data);
train_labels = cat(1, train_labels, mismatched_labels);
valid_set(:, : , mismatch_ind) = [];
validation_labels(mismatch_ind, :) = [];
validation_set_size = size(valid_set, 3);
training_set_size = size(train_set, 3);
end

save('best_classifiers.mat', 'best_classifiers');
save('weak_classifiers.mat', 'weak_classifiers');
save('face_size.mat','face_size' );

