use std::ops::{Bound, Range,RangeBounds, RangeTo};

pub(crate) fn try_range<R>(range: R, bounds: RangeTo<usize>) -> Option<Range<usize>>
where
    R: RangeBounds<usize>,
{
    let len = bounds.end;
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1)?,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    if start > end || end > len {
        return None;
    }
    Some(start..end)
}