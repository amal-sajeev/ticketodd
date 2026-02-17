(function(){
  /* ===== Scaling ===== */
  var deck = document.getElementById('deck');
  function scaleDeck(){
    var sw = window.innerWidth, sh = window.innerHeight;
    var scale = Math.min(sw / 1280, sh / 720);
    deck.style.transform = 'scale(' + scale + ')';
    deck.style.marginLeft = ((sw - 1280 * scale) / 2) + 'px';
    deck.style.marginTop = ((sh - 720 * scale) / 2) + 'px';
  }
  scaleDeck();
  window.addEventListener('resize', scaleDeck);

  /* ===== Slide Engine ===== */
  var slides = document.querySelectorAll('.slide');
  var total = slides.length;
  var progressFill = document.getElementById('progressFill');
  var counter = document.getElementById('counter');
  var current = 0;
  var animating = false;

  function updateUI(){
    progressFill.style.width = ((current + 1) / total * 100) + '%';
    counter.textContent = (current + 1) + ' / ' + total;
  }

  function goTo(idx){
    if(idx < 0 || idx >= total || idx === current || animating) return;
    animating = true;
    var dir = idx > current ? 1 : -1;
    var oldSlide = slides[current];
    var newSlide = slides[idx];

    // Remove any leftover transition classes
    oldSlide.classList.remove('enter-left','enter-right','exit-left','exit-right');
    newSlide.classList.remove('enter-left','enter-right','exit-left','exit-right');

    // Position new slide at entry point
    newSlide.classList.add(dir > 0 ? 'enter-right' : 'enter-left');
    newSlide.style.opacity = '0';
    newSlide.offsetHeight; // force reflow

    // Exit old slide
    oldSlide.classList.remove('active');
    oldSlide.classList.add(dir > 0 ? 'exit-left' : 'exit-right');

    // Enter new slide
    newSlide.classList.remove('enter-left','enter-right');
    newSlide.classList.add('active');
    newSlide.style.opacity = '';

    current = idx;
    updateUI();

    // Trigger counter animations on new slide
    animateCounters(newSlide);
    triggerSVGAnimations(newSlide);

    setTimeout(function(){
      oldSlide.classList.remove('exit-left','exit-right');
      oldSlide.style.opacity = '0';
      animating = false;
    }, 550);
  }

  function next(){ goTo(current + 1); }
  function prev(){ goTo(current - 1); }

  /* ===== Keyboard ===== */
  document.addEventListener('keydown', function(e){
    if(e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown'){ e.preventDefault(); next(); }
    if(e.key === 'ArrowLeft' || e.key === 'PageUp'){ e.preventDefault(); prev(); }
    if(e.key === 'Home'){ e.preventDefault(); goTo(0); }
    if(e.key === 'End'){ e.preventDefault(); goTo(total - 1); }
    if(e.key === 'f' || e.key === 'F'){
      if(!document.fullscreenElement) document.documentElement.requestFullscreen();
      else document.exitFullscreen();
    }
  });

  /* ===== Nav Buttons ===== */
  document.getElementById('btnNext').addEventListener('click', next);
  document.getElementById('btnPrev').addEventListener('click', prev);

  /* ===== Touch Swipe ===== */
  var touchStartX = 0;
  document.addEventListener('touchstart', function(e){ touchStartX = e.changedTouches[0].screenX; }, {passive:true});
  document.addEventListener('touchend', function(e){
    var diff = e.changedTouches[0].screenX - touchStartX;
    if(Math.abs(diff) > 50){ diff < 0 ? next() : prev(); }
  }, {passive:true});

  /* ===== Animated Counters ===== */
  function animateCounters(slide){
    var counters = slide.querySelectorAll('[data-count-to]');
    counters.forEach(function(el){
      var target = parseInt(el.getAttribute('data-count-to'), 10);
      var duration = 1500;
      var start = performance.now();
      el.textContent = '0';
      function tick(now){
        var elapsed = now - start;
        var progress = Math.min(elapsed / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(target * eased);
        if(progress < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }

  /* ===== SVG Animation Triggers ===== */
  function triggerSVGAnimations(slide){
    // Restart SMIL animations by re-inserting SVGs
    var svgs = slide.querySelectorAll('svg.anim-trigger');
    svgs.forEach(function(svg){
      var animations = svg.querySelectorAll('animate, animateTransform, animateMotion');
      animations.forEach(function(anim){
        if(anim.beginElement) anim.beginElement();
      });
    });

    // CSS-triggered bars
    var bars = slide.querySelectorAll('[data-bar-width]');
    bars.forEach(function(bar){
      bar.style.width = '0';
      requestAnimationFrame(function(){
        bar.style.transition = 'width 1.2s cubic-bezier(.16,1,.3,1)';
        bar.style.width = bar.getAttribute('data-bar-width');
      });
    });

    // Score ring
    var rings = slide.querySelectorAll('[data-ring-pct]');
    rings.forEach(function(ring){
      var pct = parseFloat(ring.getAttribute('data-ring-pct'));
      var circ = parseFloat(ring.getAttribute('data-ring-circ') || 440);
      ring.style.strokeDashoffset = circ;
      requestAnimationFrame(function(){
        ring.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(.16,1,.3,1)';
        ring.style.strokeDashoffset = circ - (circ * pct / 100);
      });
    });
  }

  /* ===== Init ===== */
  slides.forEach(function(s, i){
    if(i !== 0){
      s.style.opacity = '0';
    }
  });
  updateUI();
  // Trigger animations on first slide
  animateCounters(slides[0]);
  triggerSVGAnimations(slides[0]);
})();
