(function () {
  const STEPS = 20;
  const CENTER = STEPS / 2;

  // Higher file indices carry less detail (positive ω); img10 is the original output.
  const frame = (scene, v) => `imgs/slider${scene}/img${STEPS - v}.jpg`;

  function initDemo() {
    const original = document.getElementById('demo-original');
    const image = document.getElementById('demo-image');
    const label = document.getElementById('demo-label');
    const slider = document.getElementById('demo-omega');
    const scenesEl = document.getElementById('demo-scenes');
    const scenes = [1, 2, 3, 4];
    let scene = scenes[0];

    function render() {
      const v = Number(slider.value);
      image.src = frame(scene, v);
      const level = Math.abs(v - CENTER);
      label.textContent = level === 0 ? 'Original' : `${v < CENTER ? 'Less' : 'More'} detail · ${level}/${CENTER}`;
    }

    function preload(s) {
      for (let v = 0; v <= STEPS; v++) new Image().src = frame(s, v);
    }

    scenes.forEach((s, i) => {
      const btn = document.createElement('button');
      btn.className = 'demo-thumb' + (i === 0 ? ' is-active' : '');
      btn.setAttribute('aria-label', `Scene ${s}`);
      btn.innerHTML = `<img src="${frame(s, CENTER)}" alt="">`;
      btn.addEventListener('click', () => {
        scene = s;
        scenesEl.querySelectorAll('.demo-thumb').forEach((b) => b.classList.toggle('is-active', b === btn));
        original.src = frame(s, CENTER);
        preload(s);
        render();
      });
      scenesEl.appendChild(btn);
    });

    slider.addEventListener('input', render);
    original.src = frame(scene, CENTER);
    render();
    preload(scene);
  }

  function initCompare() {
    document.querySelectorAll('.compare').forEach((el) => {
      const range = el.querySelector('.compare-range');
      const update = () => el.style.setProperty('--pos', `${range.value}%`);
      range.addEventListener('input', update);
      update();
    });

    const scenesEl = document.getElementById('cmp-scenes');
    const viewers = document.querySelectorAll('#cmp .compare');
    const scenes = ['comp_img1', 'comp_img2'];

    function show(dir) {
      viewers.forEach((el) => {
        const suffix = el.dataset.kind === 'less' ? 'detail-' : 'detail+';
        el.querySelector('.compare-top').src = `imgs/${dir}/original.jpg`;
        el.querySelector('.compare-base').src = `imgs/${dir}/${encodeURIComponent(suffix)}.jpg`;
      });
    }

    scenes.forEach((dir, i) => {
      const btn = document.createElement('button');
      btn.className = 'demo-thumb' + (i === 0 ? ' is-active' : '');
      btn.setAttribute('aria-label', `Scene ${i + 1}`);
      btn.innerHTML = `<img src="imgs/${dir}/original.jpg" alt="">`;
      btn.addEventListener('click', () => {
        scenesEl.querySelectorAll('.demo-thumb').forEach((b) => b.classList.toggle('is-active', b === btn));
        show(dir);
      });
      scenesEl.appendChild(btn);
    });

    show(scenes[0]);
  }

  function initNav() {
    const links = document.querySelectorAll('.topnav-links a[href^="#"]');
    const sections = [...links].map((a) => document.querySelector(a.getAttribute('href')));

    function update() {
      const y = window.scrollY + window.innerHeight * 0.35;
      let current = -1;
      sections.forEach((s, i) => { if (s.getBoundingClientRect().top + window.scrollY <= y) current = i; });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) current = sections.length - 1;
      links.forEach((a, i) => a.classList.toggle('is-active', i === current));
    }

    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  function initCopy() {
    const btn = document.getElementById('copy-bibtex');
    const text = btn.querySelector('span');
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(document.getElementById('bibtex-code').textContent).then(() => {
        text.textContent = 'Copied';
        setTimeout(() => { text.textContent = 'Copy'; }, 1500);
      });
    });
  }

  function initStars() {
    const REPO = 'itsmag11/Omegance';
    const CACHE_KEY = 'omegance-stars';
    const CACHE_MS = 60 * 60 * 1000;
    const format = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, '') + 'k' : String(n));

    function show(count) {
      document.querySelectorAll('[data-gh-stars]').forEach((el) => {
        el.querySelector('b').textContent = format(count);
        el.hidden = false;
      });
    }

    // The unauthenticated GitHub API allows 60 requests per hour per visitor IP, so cache locally.
    try {
      const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null');
      if (cached && Date.now() - cached.time < CACHE_MS) {
        show(cached.count);
        return;
      }
    } catch (e) { /* ignore malformed cache */ }

    fetch(`https://api.github.com/repos/${REPO}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data) => {
        if (typeof data.stargazers_count !== 'number') return;
        show(data.stargazers_count);
        try { localStorage.setItem(CACHE_KEY, JSON.stringify({ count: data.stargazers_count, time: Date.now() })); } catch (e) { /* storage unavailable */ }
      })
      .catch(() => {});
  }

  initDemo();
  initCompare();
  initNav();
  initCopy();
  initStars();
})();
