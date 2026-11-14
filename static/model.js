async function refreshModelStatus() {
  try {
    const status = await api('/api/model/status?station='+encodeURIComponent($('inspect').elements.station.value))
    const classifier = status.classifier
    const state = status.pending_version ? ' · 正在更新 '+status.pending_version : status.last_error ? ' · 更新失败' : ''
    const platform = status.platform
    const siteMode=status.inspection?.mode==='solder_sites'
    const recognition = classifier?.ok ? '整板识别 '+classifier.version+(siteMode ? ' · 模板点位 '+status.inspection.total_sites+' 处' : '') : siteMode ? '模板点位 '+status.inspection.total_sites+' 处' : '识别未加载'
    const primary = platform?.ok ? '平台 '+(platform.name || platform.version) : '检测 '+status.version
    $('modelStatus').textContent = primary+' · '+recognition+state
    $('modelStatus').title = [status.last_error, classifier?.last_error, platform?.last_error].filter(Boolean).join('\n') || '训练完成并加载成功后自动更新'
    $('modelStatus').classList.toggle('model-warning',Boolean(status.last_error || (classifier && !classifier.ok) || (platform && !platform.ok)))
    $('activeModelLabel').textContent = (classifier?.use_with_sites ? '整板识别 '+classifier.version : platform?.ok ? ('平台检测 '+(platform.name || platform.version)) : status.label)+(siteMode ? ' · 点位 '+status.inspection.total_sites+' 处' : classifier?.ok ? ' · '+classifier.label : '')
    $('activeModelLabel').title=classifier?.label || ''
  } catch(err) {
    $('modelStatus').textContent = '模型状态暂不可用'
    $('modelStatus').title = err.message
  }
}
if (new URLSearchParams(location.search).get('page') === 'training') window.open('/platform','_blank')
refreshModelStatus()
setInterval(refreshModelStatus,5000)

async function refreshTestNotice(){
  try{
    renderModelTestNotice(await api('/platform/api/model-testing'))
  }catch{
    // Retry on the next status poll if the training platform is restarting.
  }
}
refreshTestNotice()
setInterval(refreshTestNotice,3000)
