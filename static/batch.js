let activeBatchId = localStorage.getItem('pcbActiveBatch') || ''
let activeBatch = null, galleryFilter = 'all', pollTimer, batchUploading = false
let reviewIds = []
let workspaceBatches = [], rejectedItems = [], processFilter = 'active'

async function batchApi(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload || {}),
  })
  const body = await response.json()
  if (!response.ok) throw Error(body.error || '批量操作失败')
  return body
}

async function loadBatchList(openActive = true) {
  const batches = await api('/api/batches')
  workspaceBatches = batches
  renderProcessList()
  renderRejectedBatchFilter()
  renderBatchHistory()
  if (!batches.length) {
    activeBatchId = ''
    return
  }
  if (!batches.some(batch=>batch.id === activeBatchId)) activeBatchId = batches[0].id
  if (openActive) await openBatch(activeBatchId)
}

async function openBatch(id) {
  clearTimeout(pollTimer)
  activeBatchId = id
  localStorage.setItem('pcbActiveBatch',id)
  const batch = await api('/api/batches/'+id)
  if (activeBatchId !== id) return
  const phaseChanged = ['ready','reviewing','completed'].includes(batch.state) && (activeBatch?.id !== id || activeBatch?.state !== batch.state)
  activeBatch = batch
  $('batchDetailName').textContent = batch.name+' · '+batch.station
  renderBatch(batch)
  if (phaseChanged) refresh().catch(err=>message(err.message))
  if (phaseChanged) refreshWorkspace().catch(err=>message(err.message))
  if (['importing','running'].includes(batch.state)) {
    pollTimer = setTimeout(()=>openBatch(id).catch(err=>message(err.message)),2000)
  }
}

function renderBatch(batch) {
  $('backToProcesses').textContent = batchOrigin === 'logs' ? '← 返回批次记录' : '← 返回批次列表'
  const s = batch.summary
  $('batchEmpty').hidden = true
  $('batchContent').hidden = false
  const readonly=batch.state==='completed'
  $('archiveHint').hidden=!readonly
  $('resumeImport').hidden = readonly || (batch.state !== 'importing' && !batch.items.some(item=>item.state==='failed' && !item.record_id))
  $('resumeImport').disabled = batchUploading
  $('retryBatch').hidden = readonly || !batch.items.some(item=>item.state==='failed' && item.retryable)
  document.querySelector('.sampling').hidden=readonly
  $('drawSample').disabled = readonly || !['ready','reviewing'].includes(batch.state) || !s.auto_passed
  $('finishBatch').hidden = readonly
  $('finishBatch').disabled = !['ready','reviewing'].includes(batch.state) || batchUploading
  $('batchActions').hidden=['resumeImport','retryBatch','finishBatch'].every(id=>$(id).hidden)
  const rate = value=>value === null ? '—' : value.toFixed(1)+'%'
  $('batchNumbers').innerHTML = [
    ['已检测图片',s.completed+' / '+s.total,''],
    ['合格',s.passed,'success'],
    ['不合格',s.rejected,'rejected'],
    ['机器初判合格率',rate(s.auto_rate),''],
    ['当前合格率',rate(s.rate),'success'],
    ['失败 / 已删除',s.failed+' / '+s.deleted,''],
  ].map(([label,value,cls])=>'<div><span>'+label+'</span><strong class="'+cls+'">'+value+'</strong></div>').join('')
  const defects = Object.entries(s.defects || {})
  $('batchDefects').hidden = !defects.length
  $('batchDefects').innerHTML = defects.length
    ? '<span>整板识别不合格类型</span>'+defects.map(([label,count])=>'<b>'+esc(label)+' '+count+'</b>').join('')
    : ''
  $('sampleProgress').textContent = s.sampled ? '本次抽样 '+s.sampled+' 张 · 已复查 '+s.sample_reviewed+' 张' : '仅抽机器初判合格图片'
  renderGallery()
}

function visibleBatchItems() {
  return activeBatch.items.filter(item=>{
    if (galleryFilter === 'rejected') return item.decision === '不合格'
    if (galleryFilter === 'passed') return item.decision === '合格'
    if (galleryFilter === 'sampled') return item.sampled
    if (galleryFilter === 'pending') return item.state === 'done' && item.sampled && item.revision <= item.sample_revision
    if (galleryFilter === 'failed') return item.state === 'failed'
    return true
  }).sort((a,b)=>Number(b.decision==='不合格')-Number(a.decision==='不合格') || b.sampled-a.sampled || a.ordinal-b.ordinal)
}

function renderGallery() {
  if (!activeBatch) return
  const items = visibleBatchItems()
  const fragment = document.createDocumentFragment()
  const stateLabels = {awaiting_upload:'待上传',queued:'待检测',processing:'检测中',failed:'检测失败',deleted:'已删除'}
  for (const item of items) {
    const card = document.createElement('button')
    card.className='image-card'+(item.decision === '不合格' ? ' rejected' : '')+(item.sampled ? ' sampled' : '')
    card.dataset.ordinal = item.ordinal
    card.disabled = item.state !== 'done'
    const sampledReview = item.sampled && item.revision > item.sample_revision
    const sampleBadge = item.sampled ? '<span class="sample-badge">抽样 · '+(sampledReview ? '已复查' : '待复查')+'</span>' : ''
    const preview = item.record_id ? '<img loading="lazy" src="/api/records/'+item.record_id+'/image" alt="'+esc(item.filename)+'">' : '<span class="no-preview">'+stateLabels[item.state]+'</span>'
    const state = item.state === 'done' ? item.decision : stateLabels[item.state]
    const identification = item.classification?.is_defect
      ? ' · 整板 '+esc(item.classification.label)+' '+Math.round(item.classification.score*100)+'%'
      : (item.classification ? ' · 整板未见明显缺陷' : '')
    card.innerHTML='<div class="thumbnail">'+preview+sampleBadge+'</div><div class="image-info"><strong title="'+esc(item.filename)+'">#'+String(item.ordinal+1).padStart(3,'0')+' · '+esc(item.filename)+'</strong><span class="tag" data-state="'+esc(state)+'">'+esc(state)+'</span><small>'+ (item.state === 'done' ? '机器初判 '+esc(item.auto_decision)+identification+' · '+(item.signed ? '已复查' : item.sampled ? '抽样待复查' : item.decision === '不合格' ? '异常待复查' : '自动检测') : esc(item.error || ''))+'</small></div>'
    card.onclick=async()=>{
      try {
        reviewIds = items.filter(entry=>entry.state==='done').map(entry=>entry.record_id)
        const record = await api('/api/records/'+item.record_id)
        reviewOrigin = 'batch'
        switchPage('work')
        present(record)
        $('viewer').scrollIntoView({behavior:'smooth',block:'start'})
      } catch(err) { message(err.message) }
    }
    fragment.append(card)
  }
  $('batchGallery').replaceChildren(fragment)
  if (!items.length) $('batchGallery').innerHTML='<p class="empty">没有符合条件的图片</p>'
  document.querySelectorAll('[data-filter]').forEach(button=>button.classList.toggle('active',button.dataset.filter === galleryFilter))
}

async function updateReviewNavigation(record) {
  $('previousImage').hidden = !record.batch_id
  $('nextImage').hidden = !record.batch_id
  if (!record.batch_id) return
  try {
    if (activeBatch?.id !== record.batch_id) await openBatch(record.batch_id)
    if (currentRecordId !== record.id) return
    if (!reviewIds.includes(record.id)) {
      reviewIds = activeBatch.items.filter(item=>item.state==='done').map(item=>item.record_id)
    }
    const index = reviewIds.indexOf(record.id)
    const item = activeBatch.items.find(entry=>entry.record_id === record.id)
    $('viewingFile').textContent = (item ? '#'+String(item.ordinal+1).padStart(3,'0')+' · ' : '')+(record.filename || '')
    $('previousImage').disabled = index <= 0
    $('nextImage').disabled = index < 0 || index >= reviewIds.length-1
    async function navigate(offset) {
      try {
        const next = await api((record.exception_mode ? '/api/exceptions/' : '/api/records/')+reviewIds[index+offset])
        present(next)
        $('viewer').scrollIntoView({behavior:'smooth',block:'start'})
      } catch(err) {message(err.message)}
    }
    $('previousImage').onclick=()=>navigate(-1)
    $('nextImage').onclick=()=>navigate(1)
  } catch(err) {message(err.message)}
}

async function uploadFiles(id, files) {
  batchUploading = true
  $('importRun').disabled = true
  $('resumeImport').disabled = true
  let index = 0, networkFailures = 0
  try {
    const batch = await api('/api/batches/'+id)
    const pending = batch.items.filter(item=>['awaiting_upload','failed'].includes(item.state))
    async function uploader() {
      while (index < pending.length) {
        const item = pending[index++]
        const form = new FormData()
        form.append('image',files[item.ordinal],files[item.ordinal].name)
        try {
          const response = await fetch('/api/batches/'+id+'/items/'+item.ordinal,{method:'POST',body:form})
          // A rejected image is recorded as failed by the server and excluded from the rate.
          if (!response.ok && response.status !== 400 && response.status !== 413) networkFailures++
        } catch(err) { networkFailures++ }
      }
    }
    await Promise.all([uploader(),uploader(),uploader()])
    if (networkFailures) {
      message('有 '+networkFailures+' 张上传未完成，请点击继续上传并选择原始图片。')
      return
    }
    await batchApi('/api/batches/'+id+'/start')
    message('图片已导入，正在后台检测。可切换页面或稍后回来查看。')
  } finally {
    batchUploading = false
    $('importRun').disabled = false
    $('resumeImport').disabled = false
    await openBatch(id)
  }
}

$('batchFiles').onchange = e => {
  const files = e.target.files
  $('selectedFiles').textContent = '已选择 '+files.length+' 张图片'
}
$('batchImport').onsubmit = async e=>{
  e.preventDefault()
  if (batchUploading) return
  const files = Array.from($('batchFiles').files)
  if (!files.length || files.length > 200) {message('请选择 1 至 200 张图片');return}
  const form = e.target
  $('importRun').disabled = true
  try {
    const batch = await batchApi('/api/batches',{
      station:form.elements.station.value, name:form.elements.batch.value,
      filenames:files.map(file=>file.name),
    })
    activeBatchId = batch.id
    batchOrigin = 'processing'
    galleryFilter = 'all'
    switchPage('batch')
    await loadBatchList()
    await uploadFiles(batch.id,files)
    $('batchFiles').value=''
    $('selectedFiles').textContent='可继续导入下一个批次'
  } catch(err) {message(err.message)} finally {$('importRun').disabled=false}
}
$('resumeImport').onclick = ()=>$('resumeFiles').click()
$('resumeFiles').onchange = async e=>{
  if (!activeBatch || batchUploading) return
  const files = Array.from(e.target.files)
  e.target.value=''
  if (!files.length) return
  const byName = new Map()
  for (const file of files) {
    if (!byName.has(file.name)) byName.set(file.name,[])
    byName.get(file.name).push(file)
  }
  const ordered = activeBatch.items.map(item=>byName.get(item.filename)?.shift())
  if (files.length !== activeBatch.items.length || ordered.some(file=>!file)) {
    message('请选择该批次完整的原始图片，文件名须与导入清单一致。')
    return
  }
  try {await uploadFiles(activeBatchId,ordered)} catch(err) {message(err.message)}
}
$('retryBatch').onclick=async()=>{
  try {await batchApi('/api/batches/'+activeBatchId+'/retry');await openBatch(activeBatchId)}
  catch(err) {message(err.message)}
}
$('sampleMode').onchange=()=>{
  $('sampleValue').max=$('sampleMode').value==='percent' ? '100' : '200'
}
$('drawSample').onclick=async()=>{
  if (!activeBatchId) return
  $('drawSample').disabled=true
  try {
    await batchApi('/api/batches/'+activeBatchId+'/sample',{
      mode:$('sampleMode').value,value:Number($('sampleValue').value),
    })
    galleryFilter='sampled'
    await openBatch(activeBatchId)
    message('抽样清单已生成，请逐张打开复查。')
  } catch(err) {message(err.message)}
  finally {$('drawSample').disabled=!['ready','reviewing'].includes(activeBatch?.state) || !activeBatch?.summary.auto_passed}
}
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{
  galleryFilter=button.dataset.filter
  renderGallery()
})

const batchStateLabels = {importing:'正在导入',running:'检测中',ready:'待完成归档',reviewing:'待完成归档',completed:'已完成'}
const formatRate = value => value === null || value === undefined ? '—' : value.toFixed(1)+'%'

async function enterBatch(id, origin) {
  batchOrigin = origin
  galleryFilter = 'all'
  switchPage('batch')
  $('batchContent').hidden = true
  $('batchEmpty').hidden = false
  $('batchEmpty').textContent = '正在读取批次…'
  try {await openBatch(id)} catch(err) {message(err.message)}
}

function renderProcessList() {
  const batches = workspaceBatches.filter(batch=>{
    if (processFilter === 'active') return batch.state !== 'completed'
    if (processFilter === 'completed') return batch.state === 'completed'
    return true
  })
  $('processList').replaceChildren()
  for (const batch of batches) {
    const card = document.createElement('button')
    card.className = 'process-card'
    card.dataset.batchId = batch.id
    card.innerHTML = '<div class="process-card-heading"><strong>'+esc(batch.name)+'</strong><span class="tag">'+batchStateLabels[batch.state]+'</span></div><p>'+esc(batch.station)+' · '+date(batch.created)+'</p><progress max="'+batch.total+'" value="'+batch.processed+'"></progress><div class="process-card-counts"><span>进度 '+batch.processed+' / '+batch.total+'</span><span class="rejected">不合格 '+batch.rejected+'</span><span>合格率 <b>'+formatRate(batch.rate)+'</b></span></div><small>打开批次处理 →</small>'
    card.onclick = ()=>enterBatch(batch.id,'processing')
    $('processList').append(card)
  }
  if (!batches.length) $('processList').innerHTML='<p class="empty">'+(processFilter==='active' ? '当前没有正在处理的批次，可在“导入图片”创建新批次。' : '暂无符合条件的批次')+'</p>'
  document.querySelectorAll('[data-process-filter]').forEach(button=>button.classList.toggle('active',button.dataset.processFilter===processFilter))
}

function renderRejectedBatchFilter() {
  const selected = $('rejectedBatchFilter').value
  $('rejectedBatchFilter').replaceChildren(new Option('全部已完成批次',''))
  for (const batch of workspaceBatches.filter(item=>item.state==='completed')) {
    $('rejectedBatchFilter').append(new Option(batch.name+' · '+batch.station,batch.id))
  }
  if (workspaceBatches.some(batch=>batch.id===selected && batch.state==='completed')) $('rejectedBatchFilter').value=selected
  renderRejectedList()
}

async function loadRejectedItems() {
  rejectedItems = await api('/api/batches/rejections')
  renderRejectedList()
}

function renderRejectedList() {
  const batchId = $('rejectedBatchFilter').value
  const rows = rejectedItems.filter(item=>!batchId || item.batch_id===batchId)
  $('rejectedSummary').textContent = '共 '+rows.length+' 张不合格图片 · 批次完成后自动归档 · 序号按导入顺序'
  $('rejectedList').replaceChildren()
  for (const item of rows) {
    const row = document.createElement('div')
    row.className='rejected-row'
    row.dataset.recordId=item.record_id
    row.innerHTML='<img loading="lazy" src="/api/records/'+item.record_id+'/image" alt="'+esc(item.filename)+'"><div><strong>'+esc(item.batch_name)+' <span class="serial">#'+String(item.ordinal+1).padStart(3,'0')+'</span></strong><p>'+esc(item.filename)+'</p><small>'+esc(item.station)+' · '+item.count+' 个候选'+(item.classification?.is_defect ? ' · 整板 '+esc(item.classification.label)+' '+Math.round(item.classification.score*100)+'%' : '')+(item.completed_at ? ' · 归档于 '+date(item.completed_at) : '')+'</small></div><span class="tag" data-state="'+esc(item.decision)+'">'+esc(item.decision)+'</span><button>查看图片 →</button>'
    const actions=document.createElement('div')
    actions.className='archive-actions'
    actions.append(row.lastElementChild)
    const remove=document.createElement('button')
    remove.className='danger';remove.textContent='删除'
    remove.onclick=async()=>{
      if (!confirm('从不合格产品删除「'+item.filename+'」？质量档案中的原始批次记录会保留。')) return
      remove.disabled=true
      try{
        await api('/api/exceptions/'+item.record_id,undefined,'DELETE')
        await loadRejectedItems()
        message('图片已从不合格产品删除，原始质量档案保留。')
      }catch(error){message(error.message);remove.disabled=false}
    }
    actions.append(remove);row.append(actions)
    row.querySelector('button').onclick=async()=>{
      try {
        reviewIds=rows.map(entry=>entry.record_id)
        reviewOrigin='rejected'
        const record=await api('/api/exceptions/'+item.record_id)
        switchPage('work')
        present(record)
        $('viewer').scrollIntoView({behavior:'smooth',block:'start'})
      } catch(err) {message(err.message)}
    }
    $('rejectedList').append(row)
  }
  if (!rows.length) $('rejectedList').innerHTML='<p class="empty">暂无已完成批次的不合格图片。批次完成后会自动归档到这里。</p>'
}

function renderBatchHistory() {
  const needle=$('filter').value.trim().toLowerCase()
  const expanded = new Set(Array.from($('history').querySelectorAll('details[open]')).map(node=>node.dataset.group))
  $('history').replaceChildren()
  for (const batch of workspaceBatches.filter(item=>item.state==='completed' && (item.name+' '+item.station).toLowerCase().includes(needle))) {
    const row=document.createElement('div')
    row.className='batch-history-row'
    row.dataset.batchId=batch.id
    row.innerHTML='<div><strong>'+esc(batch.name)+'</strong><small>'+esc(batch.station)+' · '+date(batch.created)+'</small></div><span>图片 <b>'+batch.total+'</b></span><span class="success">合格 '+batch.passed+'</span><span class="rejected">不合格 '+batch.rejected+'</span><div class="history-rate"><small>合格率</small><strong>'+formatRate(batch.rate)+'</strong><small>已检测 '+batch.completed+' · 失败 '+batch.failed+'</small></div><span class="tag">'+batchStateLabels[batch.state]+'</span><button>查看批次</button>'
    row.querySelector('button').onclick=()=>enterBatch(batch.id,'logs')
    const actions=document.createElement('div')
    actions.className='archive-actions'
    actions.append(row.lastElementChild)
    const remove=document.createElement('button')
    remove.className='danger';remove.textContent='删除'
    remove.onclick=async()=>{
      if (!confirm('删除批次「'+batch.name+'」？该批次全部图片、检验记录、报告及不合格产品归档将一并删除。')) return
      remove.disabled=true
      try{
        await api('/api/batches/'+batch.id,undefined,'DELETE')
        if (activeBatchId===batch.id){activeBatchId=null;activeBatch=null;reviewIds=[]}
        if (historyRows.some(record=>record.id===currentRecordId && record.batch_id===batch.id)) currentRecordId=null
        await refresh();await refreshWorkspace()
        message('批次及对应的不合格产品归档已删除。')
      }catch(error){message(error.message);remove.disabled=false}
    }
    actions.append(remove);row.append(actions)
    $('history').append(row)
  }
  // Records created before batch import was added are grouped by station and batch name.
  const legacy = new Map()
  for (const record of historyRows.filter(item=>!item.batch_id)) {
    const key=JSON.stringify([record.station,record.batch])
    if (!legacy.has(key)) legacy.set(key,[])
    legacy.get(key).push(record)
  }
  for (const [key,records] of legacy) {
    const first=records[0]
    if (!(first.batch+' '+first.station).toLowerCase().includes(needle)) continue
    const passed=records.filter(record=>record.decision==='合格').length
    const rejected=records.filter(record=>record.decision==='不合格').length
    const rate=passed+rejected ? 100*passed/(passed+rejected) : null
    const group=document.createElement('details')
    group.className='legacy-batch'
    group.dataset.group=key
    group.open=expanded.has(key)
    group.innerHTML='<summary><strong>'+esc(first.batch)+'</strong><span>'+esc(first.station)+' · 图片 '+records.length+'</span><span class="success">合格 '+passed+'</span><span class="rejected">不合格 '+rejected+'</span><strong>合格率 '+formatRate(rate)+'</strong><small>展开图片</small></summary>'
    const images=document.createElement('div')
    images.className='legacy-images'
    images.replaceChildren(...records.map(recordRow))
    group.append(images)
    $('history').append(group)
  }
  if (!$('history').children.length) $('history').innerHTML='<p class="empty">'+(needle?'没有匹配的批次':'尚无批次检验记录')+'</p>'
}

async function refreshWorkspace() {
  const results=await Promise.allSettled([loadBatchList(false),loadRejectedItems()])
  const error=results.find(result=>result.status==='rejected')
  if (error) throw error.reason
}

$('backToProcesses').onclick=()=>{
  switchPage(batchOrigin)
  refreshWorkspace().catch(err=>message(err.message))
}
document.querySelectorAll('[data-process-filter]').forEach(button=>button.onclick=()=>{
  processFilter=button.dataset.processFilter
  renderProcessList()
})
$('rejectedBatchFilter').onchange=renderRejectedList
$('finishBatch').onclick=async()=>{
  if (!activeBatchId || batchUploading) return
  const id = activeBatchId
  $('finishBatch').disabled = true
  try {
    await batchApi('/api/batches/'+id+'/finish')
    await openBatch(id)
    await refresh()
    await refreshWorkspace()
    switchPage('logs')
    message('批次已完成，不合格图片已自动存入不合格产品，完整批次已归入质量档案。')
  } catch(err) {
    message(err.message)
    await openBatch(id).catch(()=>{})
  }
}

refreshWorkspace().catch(err=>message(err.message))
setInterval(()=>{
  if (!document.hidden && ['processing','rejected','logs'].includes(currentPage)) {
    refreshWorkspace().catch(err=>message(err.message))
  }
},3000)
