# Note: it's may better as function rather than class
# pylint: disable=arguments-differ
"""Matchers for target assignment.
Matchers are commonly used in object-detection for anchor-groundtruth matching.
The matching process is a prerequisite to training target assignment.
Matching is usually not required during testing.
"""
from __future__ import absolute_import
import torch
from torch import nn


class CompositeMatcher(nn.Module):
    """A Matcher that combines multiple strategies.

    Parameters
    ----------
    matchers : list of Matcher
        Matcher is a Block/HybridBlock used to match two groups of boxes
    """

    def __init__(self, matchers):
        super(CompositeMatcher, self).__init__()
        assert len(matchers) > 0, "At least one matcher required."
        for matcher in matchers:
            assert isinstance(matcher, nn.Module)
        self._matchers = nn.ModuleList()
        for m in matchers:
            self._matchers.append(m)

    def forward(self, x):
        matches = [matcher(x) for matcher in self._matchers]
        return self._compose_matches(matches)

    def _compose_matches(self, matches):
        """Given multiple match results, compose the final match results.
        The order of matches matters. Only the unmatched(-1s) in the current
        state will be substituted with the matching in the rest matches.

        Parameters
        ----------
        matches : list of NDArrays
            N match results, each is an output of a different Matcher

        Returns
        -------
         one match results as (B, N, M) NDArray
        """
        result = matches[0]
        for match in matches[1:]:
            result = torch.where(result > -0.5, result, match)
        return result


class BipartiteMatcher(nn.Module):
    """A Matcher implementing bipartite matching strategy.

    Parameters
    ----------
    threshold : flopython -m torch.distributed.launch --nproc_per_node=$NGPUSat
        Threshold used to ignore invalid paddings
    is_ascend : bool
        Whether sort matching order in ascending order. Default is False.
    eps : float
        Epsilon for floating number comparison
    share_max : bool, default is True
        The maximum overlap between anchor/gt is shared by multiple ground truths.
        We recommend Fast(er)-RCNN series to use ``True``, while for SSD, it should
        defaults to ``False`` for better result.
    """

    def __init__(self, eps=1e-12, share_max=True):
        super(BipartiteMatcher, self).__init__()
        self._eps = eps
        self._share_max = share_max

    def forward(self, x):
        """BipartiteMatching

        Parameters:
        ----------
        x : NDArray or Symbol
            IOU overlaps with shape (N, M), batching is supported.

        """
        match = self._bipartite_matching(x)
        # make sure if iou(a, y) == iou(b, y), then b should also be a good match
        # otherwise positive/negative samples are confusing
        # potential argmax and max
        pargmax = x.argmax(dim=-1, keepdim=True)  # (B, num_anchor, 1)
        maxs, _ = x.max(dim=-2, keepdim=True)  # (B, 1, num_gt)
        if self._share_max:
            mask = (x + self._eps) >= maxs  # (B, num_anchor, num_gt)
            mask = torch.sum(mask, dim=-1, keepdim=True)  # (B, num_anchor, 1)
        else:
            pmax = torch.gather(x, -1, pargmax)  # (B, num_anchor, 1)
            mask = (pmax + self._eps) >= maxs  # (B, num_anchor, num_gt)
            mask = torch.gather(mask, -1, pargmax)  # (B, num_anchor, 1)
        new_match = torch.where(mask > 0, pargmax, torch.ones_like(pargmax) * -1)
        new_match[match] = torch.arange(0, match.numel(), dtype=match.dtype, device=match.device).unsqueeze(1)
        return new_match.squeeze()

    @staticmethod
    def _bipartite_matching(ious):
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = ious.max(1, keepdim=True)
        # [1,num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = ious.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
        # TODO refactor: index  best_prior_idx with long tensor
        # ensure every gt matches with its prior of max overlap
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j
        return best_truth_idx


class MaximumMatcher(nn.Module):
    """A Matcher implementing maximum matching strategy.

    Parameters
    ----------
    threshold : float
        Matching threshold.

    """

    def __init__(self, threshold):
        super(MaximumMatcher, self).__init__()
        self._threshold = threshold

    def forward(self, x):
        argmax = torch.argmax(x, -1, True)
        match = torch.where(x.gather(-1, argmax) >= self._threshold, argmax,
                            torch.ones_like(argmax) * -1)
        return match.squeeze()


if __name__ == '__main__':
    a = torch.Tensor([[0.1, 0.2], [0.3, .2], [0.5, 0.6], [0.4, 0.3]])
    matcher = BipartiteMatcher()
    out = matcher(a)
    print(out)
    # mather = MaximumMatcher(1.2)
    # print(mather(a))
