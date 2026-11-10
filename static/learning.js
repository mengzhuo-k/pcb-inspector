const $=id=>document.getElementById(id);
const base=location.pathname.startsWith("/training")?"/training":"";
if(new URLSearchParams(location.search).get("embedded")==="1")document.documentElement.classList.add("embedded");
const svgNS="http://www.w3.org/2000/svg";
let selected, detail, boxes=[], sites=[], width,height,start,drag,refreshing=false;
async function api(path,options) {
  const r=await fetch(base+path,options);
  if(!r.ok){const e=await r.json().catch(()=>({}));throw Error(e.error||("请求失败 "+r.status));}
  return r.json();
}
function node(tag,text,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n;}
function action(text,fn){const b=node("button",text);b.onclick=()=>Promise.resolve(fn()).catch(e=>$("msg").textContent=e.message);return b;}
async function deleteSample(sample){
  if(!confirm("删除“"+sample.name+"”及其标注？\n该图片将退出后续训练，已生成的模型和训练快照仍会保留。"))return;
  try{
    await api("/api/sample/"+sample.id,{method:"DELETE",headers:{"Content-Type":"application/json"},body:JSON.stringify({rev:sample.rev})});
    if(selected===sample.id)$("editor").close();
    $("msg").textContent="已删除“"+sample.name+"”，后续训练将使用剩余样本。";
  }finally{await refresh();}
}
async function refresh(){
  if(refreshing)return;
  refreshing=true;
  try{
    const {samples,jobs,inference}=await api("/api/state");
    $("count").textContent=samples.length+" 张 · "+samples.filter(x=>!x.ready).length+" 张待标注";
    $("trainingState").textContent=jobs[0]?.state||"等待首次训练";
    const classifier=inference?.classifier;
    if(classifier?.ok){
      const metrics=classifier.metrics||{};
      const labels=Object.values(classifier.labels||{}).join(" / ");
      $("classifierState").textContent=classifier.version+" · 已加载";
      const box=node("div","","line");
      box.append(node("strong","SMD DefDB 七类 · "+classifier.classes.length+" 类"),
        node("small","类别："+labels),
        node("small","独立测试 top1 "+(metrics.top1!=null?(100*metrics.top1).toFixed(1)+"%":"—")
          +" · top5 "+(metrics.top5!=null?(100*metrics.top5).toFixed(1)+"%":"—")
          +" · 缺陷判定阈值 "+(100*classifier.threshold).toFixed(0)+"%"),
        node("small",""+(metrics.defect_precision!=null?"整板缺陷 P "+(100*metrics.defect_precision).toFixed(1)+"% · R "+(100*metrics.defect_recall).toFixed(1)+"%":"指标暂不可读")),
        node("small","数据集 DOI 10.5281/zenodo.20001570 · 权重 "+classifier.sha256.slice(0,12)+"…"));
      $("classifier").replaceChildren(box);
    }else{
      $("classifierState").textContent="未加载";
      $("classifier").replaceChildren(node("p",classifier?.last_error||"未找到缺陷分类权重，漏焊检测仍可使用。","line"));
    }
    $("samples").replaceChildren(...samples.map(s=>{
      const r=node("div","","line"), img=new Image();img.src=base+"/api/image/"+s.id;img.loading="lazy";img.alt="";
      r.append(img,node("strong",s.name),node("small",s.split==="val"?"验证":"训练"),node("span",s.ready?s.count+" 个框":"待标注"),action(s.ready?"查看 / 修订":"标注",()=>openEditor(s.id)),action("删除",()=>deleteSample(s)));return r;
    }));
    $("versions").replaceChildren(...jobs.map(j=>{
      const r=node("div","","line");r.append(node("strong","v"+j.id+" · "+j.state),node("small","继承 "+j.parent+" · 训练 "+j.ntrain+" 张 / 验证 "+j.nval+" 张"),node("small",j.note));
      if(inference?.training_version===j.id)r.append(node("span","检测正在使用","deployed"));
      if(j.state==="完成"){const a=node("a","下载权重","action");a.href=base+"/api/model/"+j.id;r.append(a);}
      if(j.state==="失败"&&j.id===jobs[0].id)r.append(action("重试",async()=>{await api("/api/retry",{method:"POST"});await refresh();}));
      return r;
    }));
  }finally{refreshing=false;}
}
const hints={normal:"上传即作为无漏接样本加入队列。",same:"仅用于 23 个中央孔位和 3 处焊盘全部漏接的同板型。定位失败的图片会转入待标注。",partial:"上传后点击“标注”，选中实际漏接的位置，再保存。"};
$("mode").onchange=()=>$("hint").textContent=hints[$("mode").value];$("mode").onchange();
let busy=false;
async function upload(files){
  if(busy||!files.length)return;busy=true;
  const mode=$("mode").value, split=$("split").value, lines=[];
  try{
    for(const f of files){
      $("msg").textContent="上传中 · "+f.name;
      const data=new FormData();data.append("image",f);data.append("mode",mode);data.append("split",split);
      try{const r=await api("/api/upload",{method:"POST",body:data});lines.push(f.name+" · "+(r.duplicate?"已存在":r.ready?"已加入队列":"待标注"));}
      catch(e){lines.push(f.name+" · "+e.message);}
    }
    $("msg").textContent=lines.join("\n");await refresh();
  }finally{busy=false;$("file").value="";}
}
$("file").onchange=e=>upload(Array.from(e.target.files));
$("drop").onkeydown=e=>{if(e.target===$("drop")&&["Enter"," "].includes(e.key)){e.preventDefault();$("file").click();}};
$("drop").ondragover=e=>{e.preventDefault();$("drop").classList.add("active");};
$("drop").ondragleave=()=>$("drop").classList.remove("active");
$("drop").ondrop=e=>{e.preventDefault();$("drop").classList.remove("active");upload(Array.from(e.dataTransfer.files));};
function svg(tag,attrs){const n=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;}
async function openEditor(id){
  const d=await api("/api/sample/"+id);
  const img=new Image();img.src=base+"/api/image/"+id;await img.decode();
  selected=id;detail=d;boxes=d.boxes;sites=d.sites;width=img.naturalWidth;height=img.naturalHeight;
  $("canvas").setAttribute("viewBox","0 0 "+width+" "+height);
  $("sampleName").textContent=d.name;$("editMsg").textContent="";draw();$("editor").showModal();
}
const same=(a,b)=>a.every((v,i)=>Math.abs(v-b[i])<1e-6);
function rect(b,on){
  const [x,y,r,d]=b;return svg("rect",{x:x*width,y:y*height,width:(r-x)*width,height:(d-y)*height,
    fill:on?"#ffb64022":"#6aeaff15",stroke:on?"#ffb640":"#6aeaff","stroke-width":2,"vector-effect":"non-scaling-stroke","stroke-dasharray":on?"none":"4 3"});
}
function draw(){
  const c=$("canvas");c.replaceChildren(svg("image",{href:base+"/api/image/"+selected,width,height}));
  for(const s of sites)if(!boxes.some(b=>same(s,b)))c.append(rect(s,false));
  for(const b of boxes)c.append(rect(b,true));
  $("boxCount").textContent=boxes.length+" 个漏接框";
}
function point(e){
  const p=new DOMPoint(e.clientX,e.clientY).matrixTransform($("canvas").getScreenCTM().inverse());
  return [Math.min(1,Math.max(0,p.x/width)),Math.min(1,Math.max(0,p.y/height))];
}
function bounds(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];}
$("canvas").onpointerdown=e=>{if(e.button!==0)return;start=point(e);$("canvas").setPointerCapture(e.pointerId);};
$("canvas").onpointermove=e=>{if(!start)return;drag?.remove();drag=rect(bounds(start,point(e)),true);$("canvas").append(drag);};
$("canvas").onpointercancel=()=>{start=null;drag?.remove();drag=null;};
$("canvas").onpointerup=e=>{
  if(!start)return;
  const p=point(e),b=bounds(start,p);start=null;drag=null;
  if((b[2]-b[0])*width>4&&(b[3]-b[1])*height>4)boxes.push(b);
  else{const hit=r=>p[0]>=r[0]&&p[0]<=r[2]&&p[1]>=r[1]&&p[1]<=r[3];
    const idx=boxes.findIndex(hit);if(idx>=0)boxes.splice(idx,1);else{const s=sites.find(hit);if(s)boxes.push(s);}
  }draw();
};
$("close").onclick=()=>$("editor").close();
$("save").onclick=async()=>{
  $("save").disabled=true;
  try{await api("/api/sample/"+selected,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({boxes,rev:detail.rev})});
    $("editor").close();$("msg").textContent="标注已保存，训练队列已更新。";await refresh();
  }catch(e){$("editMsg").textContent=e.message;}finally{$("save").disabled=false;}
};
function poll(){refresh().catch(e=>$("msg").textContent=e.message);}
poll();setInterval(poll,5000);
