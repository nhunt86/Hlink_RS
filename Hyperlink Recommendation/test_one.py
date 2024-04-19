import tensorflow as tf
import os
import sys
import numpy as np
import math, random
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *

class NGCF(object):
    def __init__(self, data_config, pretrain_data):#,data_generator=None
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
       # self.data_generator = data_generator
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        
        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))
        
        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        # mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

# def generate_negative_samples(user, num_samples):
#     # Xác định phạm vi mục tiêu, ví dụ, từ 1 đến N (trong đó N là số lượng mục)
#     num_items = data_generator.n_items  # Thay N bằng số lượng mục thực tế
#     target_items = set(range(1, num_items + 1))

#     # Lấy các mục (items) mà người dùng (user) đã tương tác trong tập huấn luyện
#     interacted_items = list(data_generator.train_set[user])

#     # Loại bỏ các mục đã tương tác khỏi phạm vi mục tiêu
#     non_interacted_items = target_items - set(interacted_items)

#     # Lựa chọn ngẫu nhiên 'num_samples' mục không tương tác từ phạm vi mục tiêu
#     negative_samples = random.sample(non_interacted_items, num_samples)

#     return negative_samples

def generate_negative_samples(user, num_samples):
    # Lấy tất cả các mục (items) trong tập dữ liệu
    # all_items = data_generator.n_items
    all_items = list(range(data_generator.n_items))

    # Lấy các mục (items) mà người dùng (user) đã tương tác trong tập huấn luyện
    interacted_items = list(data_generator.test_set[user])

    # Tạo danh sách các mục không tương tác (âm) bằng cách loại bỏ các mục đã tương tác
    non_interacted_items = [item for item in all_items if item not in interacted_items]

    # Lựa chọn ngẫu nhiên 'num_samples' mục không tương tác từ danh sách non_interacted_items
    negative_samples = random.sample(non_interacted_items, num_samples)

    return negative_samples


def calculate_mae_and_rmse(sess, model):
    users_to_test = list(data_generator.test_set.keys())

    mae_list, rmse_list = [], []

    for user in users_to_test:
        user_items = list(data_generator.test_set[user])
        users = [user] * len(user_items)
        neg_items = generate_negative_samples(user, len(user_items))  # Sửa đổi hàm này để tạo mẫu âm

        test_dict = {model.users: users, model.pos_items: user_items, model.neg_items: neg_items}

        predictions = sess.run(model.batch_ratings, feed_dict=test_dict)

        ground_truth = [data_generator.test_set[user][item] for item in user_items]
        errors = predictions - ground_truth
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        mae_list.append(mae)
        rmse_list.append(rmse)

    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)

    return avg_mae, avg_rmse

def calculate_mae_and_rmse1(sess, model):
    # Lấy dữ liệu kiểm tra từ hàm test()
    users_to_test = list(data_generator.test_set.keys())  # Dựa vào cách tạo dữ liệu kiểm tra trong hàm test()

    # Dùng dữ liệu kiểm tra để tính MAE và RMSE
    mae_list, rmse_list = [], []
    for user in users_to_test:
        user_items = list(data_generator.test_set[user])
        users = [user] * len(user_items)

        test_dict = {model.users: users, model.pos_items: user_items}

        predictions = sess.run(model.batch_ratings, feed_dict=test_dict)

        # Tính MAE và RMSE dựa trên dự đoán và giá trị thực tế
        ground_truth = [data_generator.test_set[user][item] for item in user_items]
        errors = predictions - ground_truth
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        mae_list.append(mae)
        rmse_list.append(rmse)

    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)

    return avg_mae, avg_rmse


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


def generate_recommendations(sess, model, user_id, top_k=10):
    """
    Generate recommendations for a user using the trained NGCF model.

    Parameters:
    - sess: TensorFlow session with the loaded model.
    - model: NGCF model instance.
    - user_id: User ID for which recommendations are generated.
    - top_k: Number of top recommendations to return.

    Returns:
    - recommendations: List of top-k item IDs recommended to the user.
    """

    # Generate negative samples for the user (used to rank recommendations)
    num_samples = model.n_items  # You can adjust the number of negative samples as needed
    neg_items = generate_negative_samples(user_id, num_samples)

    # Prepare input data for the user and their negative samples
    users = [user_id] * num_samples

    test_dict = {
        model.users: users,
        model.pos_items: [user_id] * num_samples,  # Positive items (not used for ranking)
        model.neg_items: neg_items
    }

    # Get predictions for the negative samples
    predictions = sess.run(model.batch_ratings, feed_dict=test_dict)

    # Rank the items based on predicted scores
    item_scores = list(enumerate(predictions))
    item_scores.sort(key=lambda x: x[1], reverse=True)

    # Select the top-k items as recommendations
    recommendations = [item_id for item_id, score in item_scores[:top_k]]

    return recommendations



def predict_items_for_user(sess, model, user_id, drop_flag=False):
    # Lấy dữ liệu và model đã khởi tạo từ môi trường
    # data_generator = model.data_generator
    # ITEM_NUM = data_generator.n_items
    
    # Xác định user_pos_test của người dùng được chỉ định
    user_pos_test = data_generator.test_set[user_id]
    
    # Xác định tất cả các item trong tập dữ liệu
    all_items = set(range(ITEM_NUM))
    
    # Xác định các item đã huấn luyện cho người dùng
    try:
        training_items = data_generator.train_items[user_id]
    except Exception:
        training_items = []
    
    # Xác định các item chưa được huấn luyện cho người dùng
    test_items = list(all_items - set(training_items))
    
    # Tính toán xếp hạng cho các item chưa được huấn luyện
    rating = sess.run(model.batch_ratings, {
        model.users: [user_id],
        model.pos_items: test_items,
        model.node_dropout: [0.] * len(eval(args.layer_size)),
        model.mess_dropout: [0.] * len(eval(args.layer_size))
    })
    
    # Xây dựng item_score từ rating
    item_score = {item: score for item, score in zip(test_items, rating[0])}
    
    # Sắp xếp item_score để lấy ra K_max_item_score
    #K_max_item_score = heapq.nlargest(len(item_score), item_score, key=item_score.get)
    K_max_item_score = heapq.nlargest(50, item_score, key=item_score.get)
    
    # In ra các items dự đoán cho người dùng
    print("Predicted items for user {}: {}".format(user_id, K_max_item_score))

def predict_items_for_user2(sess, model, user_id, drop_flag=False):
    # Lấy các thông số từ model
    n_users = model.n_users
    n_items = model.n_items
    
    # Tính toán xếp hạng cho các item chưa được huấn luyện
    rating = sess.run(model.batch_ratings, {
        model.users: [user_id],
        model.node_dropout: [0.] * model.n_layers,
        model.mess_dropout: [0.] * model.n_layers
    })
    
    # Lấy ra giá trị rating cho mỗi item
    item_ratings = rating[0]
    
    # Chọn ra 50 mục có điểm rating cao nhất
    top_items = heapq.nlargest(50, range(len(item_ratings)), key=item_ratings.__getitem__)
    
    # In ra 50 mục có điểm rating cao nhất
    for item_id in top_items:
        print("Item {}: Rating {}".format(item_id, item_ratings[item_id]))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()

    if args.pretrain == 1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    #test 1
    # ngcf_instance = NGCF(data_config=config, pretrain_data=pretrain_data)
    # # opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # init_op = ngcf_instance._init_weights()

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())
    # cur_best_pre_0 = 0.
    # sess = tf.Session()
    # sess.run(init_op)

    #test 2:
    model = NGCF(data_config=config, pretrain_data=pretrain_data)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')
    # saver.restore(sess, pretrain_path)
    #recommendation
    #predict_items_for_user(sess, model, user_id=12)

    print("----------------")
    # predict_items_for_user(sess, model, user_id=1012)
    print("----------------")
    # predict_items_for_user(sess, model, user_id=2012)
    predict_items_for_user(sess, model, user_id=2286)
    print("----------------")