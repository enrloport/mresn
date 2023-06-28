include("../ESN.jl")
# using DataFrames
# using CSV


# Cargar el conjunto de datos MNIST TRAIN
data_x, data_y = MNIST(split=:train)[:]
file_x    = "augmented_mnist_train_x_180000_2.csv"
file_y    = "augmented_mnist_train_y_180000_2.csv"
sz        = size(data_x,3)

# # Cargar el conjunto de datos MNIST TEST
# data_x , data_y  = MNIST(split=:test)[:]
# file_x    = "augmented_mnist_test_x.csv"
# file_y    = "augmented_mnist_test_y.csv"
# sz        = size(data_x,3)


function augment_img(index, data_x, angls= [-0.5, -0.25, 0.25, 0.5])
    im1 = data_x[:,:,index]'
    # im2 = imrotate(colorview(Gray, im1), angls[1], axes=(1, 2), fill=0)
    im3 = imrotate(colorview(Gray, im1), angls[2], axes=(1, 2), fill=0)
    im4 = imrotate(colorview(Gray, im1), angls[3], axes=(1, 2), fill=0)
    # im5 = imrotate(colorview(Gray, im1), angls[4], axes=(1, 2), fill=0)

    # im2_cr = im2[3:end-3, 3:end-3]
    im3_cr = im3[2:end-2, 2:end-2]
    im4_cr = im4[2:end-2, 2:end-2]
    # im5_cr = im5[3:end-3, 3:end-3]

    im1_rs = reshape(Float16.(im1),    (28*28,:) )
    # im2_rs = reshape(replace!(Float16.(im2_cr), NaN=>0.0), (28*28,:) )
    im3_rs = reshape(replace!(Float16.(im3_cr), NaN=>0.0), (28*28,:) )
    im4_rs = reshape(replace!(Float16.(im4_cr), NaN=>0.0), (28*28,:) )
    # im5_rs = reshape(replace!(Float16.(im5_cr), NaN=>0.0), (28*28,:) )

    return im1_rs, im3_rs, im4_rs#, im2_rs, im5_rs
end


function add_augmented_image(index, data_x, data_y, file_x, file_y)
    as    = augment_img(index, data_x)
    label = data_y[index]
    for img in as
        CSV.write(file_x, Tables.table(img'), append = true)
        CSV.write(file_y, Tables.table([label]), append = true)
    end
end


for index in 1:sz
    add_augmented_image(index, data_x, data_y, file_x, file_y)
end

# function test_img(index,data_x)
#     Images.Gray.(hcat([reshape(x, 28,28) for x in augment_img(index, data_x)]...))
# end

# ti = test_img(1,data_x)