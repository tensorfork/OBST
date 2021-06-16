import mesh_tensorflow as mtf
import tensorflow as tf
import typing
tf1 = tf.compat.v1

tf1.disable_v2_behavior()


def tests_wrap(build_func, run_func, mesh_shape=None, layout_rules=None, devices=None):

    if mesh_shape is None:
        mesh_shape = []
    if layout_rules is None:
        layout_rules = []
    if devices is None:
        devices = ["cpu:0"]

    default_session = tf1.get_default_session()
    if default_session is not None:
        default_session.close()

    session_config = tf1.ConfigProto()
    session_config.allow_soft_placement = True

    with tf.Graph().as_default() as tf_graph:
        with tf1.Session(config=session_config, graph=tf_graph) as sess:
            mtf_graph = mtf.Graph()
            mesh = mtf.Mesh(mtf_graph, "MESH")

            outputs, *args = build_func(mesh)

            mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, devices)
            lowering = mtf.Lowering(mtf_graph, {mesh: mesh_impl})

            outputs = [lowering.export_to_tf_tensor(outputs) for outputs in outputs]

            sess.run(tf1.global_variables_initializer())
            sess.run(lowering.copy_masters_to_slices())

            run_func(sess, outputs, args)


if __name__ == '__main__':
    def build_func(mesh):
        batch_dim = mtf.Dimension("batch", 1000)
        rows_dim = mtf.Dimension("rows", 28)
        cols_dim = mtf.Dimension("cols", 28)
        hidden_dim = mtf.Dimension("hidden", 1024)
        classes_dim = mtf.Dimension("classes", 10)

        tf_images = tf1.random.uniform(shape=[1000, 28, 28])
        images = mtf.import_tf_tensor(mesh, tf_images, shape=[batch_dim, rows_dim, cols_dim])

        w1 = mtf.get_variable(mesh, "w1", [rows_dim, cols_dim, hidden_dim])
        w2 = mtf.get_variable(mesh, "w2", [hidden_dim, classes_dim])

        hidden = mtf.relu(mtf.einsum([images, w1], output_shape=[batch_dim, hidden_dim]))
        logits = mtf.einsum([hidden, w2], output_shape=[batch_dim, classes_dim])

        return [logits], None

    def run_func(sess, outputs, *args):
        print(sess.run(outputs)[0].shape)


    tests_wrap(build_func, run_func)