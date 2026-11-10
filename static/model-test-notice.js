const testNoticeOpenedAt = Date.now() / 1000
let lastTestNotice = ''

function renderModelTestNotice(job){
  const notice = document.getElementById('modelTestNotice')
  if (job?.state === 'running'){
    if (lastTestNotice === job.id && !notice.hidden) return
    notice.hidden = false
    notice.textContent = '新模型正在测试，识别系统继续使用当前模型。'
    lastTestNotice = job.id
    return
  }
  // A small test set can finish between two status polls.
  if (job && job.testing_started >= testNoticeOpenedAt && lastTestNotice !== job.id){
    lastTestNotice = job.id
    notice.textContent = job.state === 'done' ? '新模型测试已结束。' : '新模型测试已中断。'
    notice.hidden = false
    return
  }
  notice.hidden = true
}
