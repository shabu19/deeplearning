from fuse_validate_model import ResearchModels
from fuse_validate_data import DataSet

def test_1epoch_fuse(class_limit=None, n_snip=5, opt_flow_len=10, saved_model=None, saved_spatial_weights=None, saved_temporal_weights=None, image_shape=(224, 224), original_image_shape=(341, 256), batch_size=128, fuse_method='average'):

    # get the data
    data = DataSet(class_limit=class_limit, image_shape=image_shape, original_image_shape=original_image_shape, n_snip=n_snip, opt_flow_len=opt_flow_len, batch_size=batch_size)

    # get the validation generator
    val_generator = data.validation_generator()
    steps = data.n_batch

    # get the model
    two_stream_fuse = ResearchModels(nb_classes=len(data.classes), n_snip=n_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, saved_model=saved_model, saved_temporal_weights=saved_temporal_weights, saved_spatial_weights=saved_spatial_weights)

    # evaluate
    two_stream_fuse.model.fit(x=val_generator, steps_per_epoch=steps, max_queue_size=1)

def main():
    saved_spatial_weights = 'C:/Users/User/Documents/PyCharm_Projects/two-stream-action-recognition-keras-master/out/checkpoints/cp-spatial/071.hdf5'
    saved_temporal_weights = 'C:/Users/User/Desktop/temporal-result-100-128-0.5/205.hdf5'
    class_limit = 100
    n_snip = 19                         # number of chunks used for each video
    opt_flow_len = 10                   # number of optical flow frames used
    image_shape=(224, 224)
    original_image_shape=(341, 256)
    batch_size = 128
    fuse_method = 'average'

    test_1epoch_fuse(class_limit=class_limit, n_snip=n_snip, opt_flow_len=opt_flow_len, saved_spatial_weights=saved_spatial_weights, saved_temporal_weights=saved_temporal_weights, image_shape=image_shape, original_image_shape=original_image_shape, batch_size=batch_size, fuse_method=fuse_method)

if __name__ == '__main__':
    main()
