
def predict_items_for_user(sess, model, user_id, drop_flag=False):
    # Lấy dữ liệu và model đã khởi tạo từ môi trường
    data_generator = model.data_generator
    ITEM_NUM = data_generator.n_items
    
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
    K_max_item_score = heapq.nlargest(len(item_score), item_score, key=item_score.get)
    
    # In ra các items dự đoán cho người dùng
    print("Predicted items for user {}: {}".format(user_id, K_max_item_score))