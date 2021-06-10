import math
import random
import time

import torch

def rel_close(a, b, threshold):
    if b > a:
        a, b = b, a
    a += 1e-9
    b += 1e-9
    return (a/b - 1) < threshold

def constant_tuple_numels(c1, c2):
    return c1, c2

def random_tuple_numels(high):
    tn1 = random.randint(1, high)
    tn2 = random.randint(1, high)
    return tn1, tn2

def make_positive_pairs(num_tuples, num_positives, max_positives):

    if num_positives == 'random':
        tuple_numels = lambda: random_tuple_numels(max_positives)
    else:
        tuple_numels = lambda: constant_tuple_numels(*num_positives)

    positive_pairs = []
    numel_1 = 0
    numel_2 = 0
    for tuple_idx in range(num_tuples):
        tuple_numel_1, tuple_numel_2 = tuple_numels()

        for i1 in range(tuple_numel_1):
            for i2 in range(tuple_numel_2):
                pair = [numel_1+i1, numel_2+i2]
                positive_pairs.append(pair)

        numel_1 += tuple_numel_1
        numel_2 += tuple_numel_2

    permutation = list(range(numel_2))
    random.shuffle(permutation)

    positive_pairs = list(map(
        lambda p: (p[0], permutation[p[1]]),
        positive_pairs))
    
    positive_pairs = torch.tensor(positive_pairs, dtype=torch.long)
    return positive_pairs, (numel_1, numel_2)

def test_naive_hardest_triplet_margin_loss():
    import triplet_margin_loss as tml

    scores = torch.tensor([
        [-0.1,  0.2, -0.6, -0.5],
        [ 0.9,  0.4, -0.1,  0.4],
        [-0.2, -0.3,  0.3, -0.0],
        [ 0.7, -0.5, -0.5,  0.5],
    ])

    positive_indices = torch.tensor([
        [0, 0],
        [1, 3],
        [2, 2],
        [3, 1],
    ])
    targets = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ])

    margin = 0.2

    row_violations = torch.tensor([
        [ 0.0,  0.5, -0.0, -0.0],
        [ 0.7,  0.2, -0.0,  0.0],
        [-0.0, -0.0,  0.0, -0.0],
        [ 1.4,  0.0,  0.2,  1.2],
    ])
    col_violations = torch.tensor([
        [ 0.0,  0.7, -0.9, -0.9], 
        [ 1.0,  0.9, -0.4,  0.0], 
        [-0.1,  0.2,  0.0, -0.4], 
        [ 0.8,  0.0, -0.8,  0.1], 
    ])

    row_losses_sum = torch.tensor([
        (0.5 + 0.0 + 0.0) / 3,
        (0.7 + 0.2 + 0.0) / 3,
        (0.0 + 0.0 + 0.0) / 3,
        (1.4 + 0.2 + 1.2) / 3,
    ])
    row_losses_half = torch.tensor([
        (0.5 + 0.0) / 2,
        (0.7 + 0.2) / 2,
        (0.0 + 0.0) / 2,
        (1.4 + 1.2) / 2,
    ])
    row_losses_max = torch.tensor([
        (0.5) / 1,
        (0.7) / 1,
        (0.0) / 1,
        (1.4) / 1,
    ])

    col_violations = torch.tensor([
        [ 0.0,  0.9, -0.0, -0.0], 
        [ 1.2,  1.1, -0.0,  0.0], 
        [ 0.1,  0.4,  0.0, -0.0], 
        [ 1.0,  0.0, -0.0,  0.3], 
    ])
    col_losses_sum = torch.tensor([
        (1.2 + 1.0 + 0.1) / 3,
        (1.1 + 0.9 + 0.4) / 3,
        (0.0 + 0.0 + 0.0) / 3,
        (0.3 + 0.0 + 0.0) / 3,
    ])
    col_losses_half = torch.tensor([
        (1.2 + 1.0) / 2,
        (1.1 + 0.9) / 2,
        (0.0 + 0.0) / 2,
        (0.3 + 0.0) / 2,
    ])
    col_losses_max = torch.tensor([
        (1.2) / 1,
        (1.1) / 1,
        (0.0) / 1,
        (0.3) / 1,
    ])

    start_idx, end_idx = 0, scores.size(0)+1

    hardest_fraction_losses = {
        1.0: (row_losses_sum, col_losses_sum),
        0.5: (row_losses_half, col_losses_half),
        0.0: (row_losses_max, col_losses_max),
    }

    scores = scores[start_idx:end_idx]
    targets = targets[start_idx:end_idx]

    for hardest_fraction in hardest_fraction_losses:
        row_losses, col_losses = hardest_fraction_losses[hardest_fraction]

        hard_coded_row_loss = row_losses.mean()/margin
        hard_coded_col_loss = col_losses.mean()/margin

        naive_row_loss = tml.naive_hardest_fraction_triplet_margin_loss(
            scores, targets, margin, hardest_fraction)
        naive_col_loss = tml.naive_hardest_fraction_triplet_margin_loss(
            scores.t(), targets.t(), margin, hardest_fraction)

        assert hard_coded_row_loss == naive_row_loss
        assert hard_coded_col_loss == naive_col_loss


def test_triplet_margin_losses():
    import triplet_margin_loss as tml
    import retrieval_metrics

    def sparse_loss_fn(scores, targets, indices, margin, hardest_fraction):
        def loss_fn(preds, targets):
            return tml.anchor_hardest_fraction_triplet_margin_loss(
                    preds, targets, margin, hardest_fraction)

        loss = retrieval_metrics.apply_retrieval_metric(
            scores.reshape(-1), targets.reshape(-1), indices, loss_fn).mean()
        return loss

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    num_tuples = 32
    num_trials = 30
    max_positives = 5

    functions = {
        'naive'     : lambda p,t,i,m,h:
            tml.naive_hardest_fraction_triplet_margin_loss(p,t,m,h),
        'anchor'    : lambda p,t,i,m,h: 
            sparse_loss_fn(p,t,i,m,h),
        'chunked'   : lambda p,t,i,m,h:
            tml.chunked_hardest_fraction_triplet_margin_loss(p,t,m,h),
        'vectorized': lambda p,t,i,m,h: 
            tml.vectorized_hardest_fraction_triplet_margin_loss(p,t,m,h),
        'single'    : lambda p,t,i,m,h: 
            tml.single_hardest_fraction_triplet_margin_loss(p,t,m,h),
        'constant'  : lambda p,t,i,m,h: 
            tml.constant_hardest_fraction_triplet_margin_loss(p,t,m,h),
        'hardest'   : lambda p,t,i,m,h: 
            tml.hardest_triplet_margin_loss(p,t,m),
    }

    #(Single|Constant|Multiple)Positives_Hardest[Fraction]
    group_parameters = {
        'SP_HF': ('single', 'random'),
        'CP_HF': ('constant', 'random'),
        'MP_HF': ('random', 'random'),
        'MP_H' : ('random', 0.0)
    }
    group_functions = {
        'SP_HF': ['naive', 'anchor', 'chunked', 'vectorized', 'single', 'constant'],
        'CP_HF': ['naive', 'anchor', 'chunked', 'vectorized', 'constant'],
        'MP_HF': ['naive', 'anchor', 'chunked', 'vectorized'],
        'MP_H' : ['naive', 'anchor', 'chunked', 'vectorized', 'hardest'],
    }

    error_threshold = 1e-6

    for group_name in group_parameters:
        print(f"Starting {group_name}.")
        num_positives_param, hardest_fraction_param = group_parameters[group_name]

        if num_positives_param == 'single':
            num_positives_fn = lambda: [1,1]
        elif num_positives_param == 'constant':
            num_positives_fn = lambda: list(random_tuple_numels(max_positives))
        elif num_positives_param == 'random':
            num_positives_fn = lambda: 'random'

        if hardest_fraction_param == 'random':
            hardest_fraction_fn = lambda: random.uniform(0,1)
        else:
            hardest_fraction_fn = lambda: hardest_fraction_param

        function_names = group_functions[group_name]
        function_times = {
            function_name: 0.0
            for function_name in function_names
        }

        for i in range(num_trials):
            positive_pairs, preds_size = make_positive_pairs(
                num_tuples, 
                num_positives_fn(),
                max_positives)
            preds = torch.distributions.uniform.Uniform(-1, 1).sample(preds_size)
            targets, indices_1, indices_2 = retrieval_metrics.positive_sparse2dense(
                positive_pairs, list(preds_size))
            targets = targets.view(preds_size)

            indices_2 = indices_2.view(preds_size).t().reshape(-1)

            margin = random.uniform(0, 1)
            hardest_fraction = hardest_fraction_fn()

            function_values = {}
            for function_name in function_names:
                start_time = time.time()
                value = functions[function_name](
                    preds, targets, indices_1, margin, hardest_fraction)
                function_values[function_name] = value
                function_times[function_name] += time.time() - start_time
            reference_value = function_values['naive']
            for function_name, function_value in function_values.items():
                if not rel_close(reference_value, function_value, error_threshold):
                    print(f"{group_name}: n°{i}: 122: {function_name}: {reference_value} vs {function_value}")

            function_values = {}
            for function_name in function_names:
                start_time = time.time()
                value = functions[function_name](
                    preds.t(), targets.t(), indices_2, margin, hardest_fraction)
                function_values[function_name] = value
                function_times[function_name] += time.time() - start_time
            reference_value = function_values['naive']
            for function_name, function_value in function_values.items():
                if not rel_close(reference_value, function_value, error_threshold):
                    print(f"{group_name}: n°{i}: 221: {function_name}: {reference_value} vs {function_value}")

        min_time = min(function_times.values())
        for function_name, function_time in function_times.items():
            norm_time = function_time / min_time
            print(f"{function_name}: {norm_time:.3f}")

def test_naive_recall_at_k():
    import retrieval_metrics

    scores = torch.tensor([
        [-0.1,  0.2, -0.6, -0.5],
        [ 0.9,  0.41, -0.1, 0.41],
        [-0.2, -0.3,  0.3, -0.0],
        [ 0.7, -0.52, -0.5,  0.5],
    ])

    positive_indices = torch.tensor([
        [0, 0],
        [1, 3],
        [2, 2],
        [3, 1],
    ])
    targets = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ])

    row_recall_ranks = torch.tensor([
        2,
        3,
        1,
        4,
    ])
    col_recall_ranks = torch.tensor([
        3,
        4,
        1,
        2
    ])


    for k in range(1, 4):
        hard_coded_recall = ((row_recall_ranks-1) < k).float()
        hard_coded_recall = hard_coded_recall.mean()
        naive_recall = retrieval_metrics.naive_recall_at_k(scores, targets, k)
        naive_recall = sum(naive_recall) / len(naive_recall)
        assert hard_coded_recall == naive_recall
    for k in range(1, 4):
        hard_coded_recall = ((col_recall_ranks-1) < k).float()
        hard_coded_recall = hard_coded_recall.mean()
        naive_recall = retrieval_metrics.naive_recall_at_k(scores.t(), targets.t(), k)
        naive_recall = sum(naive_recall) / len(naive_recall)
        assert hard_coded_recall == naive_recall

def test_recall_at_k():
    import triplet_margin_loss as tml
    import retrieval_metrics as rm

    def sparse_recall_fn(scores, targets, indices, k):
        def loss_fn(preds, targets):
            return rm.anchor_recall_at_k(
                    preds, targets, k)

        loss = rm.apply_retrieval_metric(
            scores.reshape(-1), targets.reshape(-1), indices, loss_fn).mean()
        return loss

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)


    num_tuples = 32
    num_trials = 30
    max_positives = 5

    functions = {
        'naive'     : lambda p,t,i,k:
            torch.mean(torch.tensor(rm.naive_recall_at_k(p,t,k))),
        'anchor'     : lambda p,t,i,k:
            torch.mean(sparse_recall_fn(p,t,i,k)),
        'vectorized'     : lambda p,t,i,k:
            torch.mean(rm.recall_at_k(p,t,k)),
    }

    #(Single|Constant|Multiple)Positives_Hardest[Fraction]
    group_parameters = {
        'SP': ('single', 'random'),
        'MP': ('random', 'random'),
    }
    group_functions = {
        'SP': ['naive', 'anchor', 'vectorized'],
        'MP': ['naive', 'anchor', 'vectorized'],
    }

    error_threshold = 1e-6

    for group_name in group_parameters:
        print(f"Starting {group_name}.")
        num_positives_param, k_param = group_parameters[group_name]

        if num_positives_param == 'single':
            num_positives_fn = lambda: [1,1]
        elif num_positives_param == 'constant':
            num_positives_fn = lambda: list(random_tuple_numels(max_positives))
        elif num_positives_param == 'random':
            num_positives_fn = lambda: 'random'

        if k_param == 'random':
            k_fn = lambda: random.randint(1,max_positives+2)
        else:
            k_fn = lambda: k_param

        function_names = group_functions[group_name]
        function_times = {
            function_name: 0.0
            for function_name in function_names
        }

        for i in range(num_trials):
            positive_pairs, preds_size = make_positive_pairs(
                num_tuples, 
                num_positives_fn(),
                max_positives)
            preds = torch.distributions.uniform.Uniform(-1, 1).sample(preds_size)
            targets, indices_1, indices_2 = rm.positive_sparse2dense(
                positive_pairs, list(preds_size))
            targets = targets.view(preds_size)

            indices_2 = indices_2.view(preds_size).t().reshape(-1)

            k = k_fn()

            function_values = {}
            for function_name in function_names:
                start_time = time.time()
                value = functions[function_name](
                    preds, targets, indices_1, k)
                function_values[function_name] = value
                function_times[function_name] += time.time() - start_time
            reference_value = function_values['naive']
            for function_name, function_value in function_values.items():
                if not rel_close(reference_value, function_value, error_threshold):
                    print(f"{group_name}: n°{i}: 122: {function_name}: {reference_value} vs {function_value}")

            function_values = {}
            for function_name in function_names:
                start_time = time.time()
                value = functions[function_name](
                    preds.t(), targets.t(), indices_2, k)
                function_values[function_name] = value
                function_times[function_name] += time.time() - start_time
            reference_value = function_values['naive']
            for function_name, function_value in function_values.items():
                if not rel_close(reference_value, function_value, error_threshold):
                    print(f"{group_name}: n°{i}: 221: {function_name}: {reference_value} vs {function_value}")

        min_time = min(function_times.values())
        for function_name, function_time in function_times.items():
            norm_time = function_time / min_time
            print(f"{function_name}: {norm_time:.3f}")



if __name__ == '__main__':
    #test_naive_hardest_triplet_margin_loss()
    #test_triplet_margin_losses()

    test_naive_recall_at_k()
    test_recall_at_k()
