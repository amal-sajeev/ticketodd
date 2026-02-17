/* Analytics JS */
let deptChartInstance = null;
let sentChartInstance = null;

if (requireAuth()) { loadAnalytics(); }

async function loadAnalytics() {
  const days = document.getElementById('period').value;
  showLoading('statsCards');
  try {
    const data = await api('GET', `/analytics?days=${days}`);
    // KPI cards
    document.getElementById('statsCards').innerHTML = `
      <div class="metric-card card-animate"><div class="metric-icon blue"><span class="icon">assignment</span></div><div class="metric-value">${data.total_grievances}</div><div class="metric-label">Total</div></div>
      <div class="metric-card card-animate"><div class="metric-icon green"><span class="icon">check_circle</span></div><div class="metric-value">${data.self_resolved}</div><div class="metric-label">Self-Resolved</div></div>
      <div class="metric-card card-animate"><div class="metric-icon purple"><span class="icon">auto_awesome</span></div><div class="metric-value">${data.ai_drafted}</div><div class="metric-label">AI Drafted</div></div>
      <div class="metric-card card-animate"><div class="metric-icon red"><span class="icon">warning</span></div><div class="metric-value">${data.escalated_to_human}</div><div class="metric-label">Escalated</div></div>
      <div class="metric-card card-animate"><div class="metric-icon amber"><span class="icon">schedule</span></div><div class="metric-value">${data.avg_resolution_time}h</div><div class="metric-label">Avg Resolution</div></div>`;

    // Department chart
    const deptLabels = Object.keys(data.department_distribution).map(d => deptLabel(d));
    const deptValues = Object.values(data.department_distribution);
    if (deptChartInstance) deptChartInstance.destroy();
    deptChartInstance = new Chart(document.getElementById('deptChart'), {
      type: 'bar',
      data: {
        labels: deptLabels,
        datasets: [{ label: 'Grievances', data: deptValues,
          backgroundColor: 'rgba(26,115,232,0.6)',
          borderColor: 'rgba(26,115,232,0.9)', borderWidth: 1, borderRadius: 8 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } },
                  x: { grid: { display: false }, ticks: { maxRotation: 45 } } }
      }
    });

    // Sentiment chart
    const sentLabels = Object.keys(data.sentiment_distribution).map(s => s.charAt(0).toUpperCase() + s.slice(1));
    const sentValues = Object.values(data.sentiment_distribution);
    const sentColors = { Positive: '#1E8E3E', Neutral: '#5F6368', Negative: '#E37400', Frustrated: '#D93025' };
    if (sentChartInstance) sentChartInstance.destroy();
    sentChartInstance = new Chart(document.getElementById('sentChart'), {
      type: 'doughnut',
      data: {
        labels: sentLabels,
        datasets: [{ data: sentValues,
          backgroundColor: sentLabels.map(l => sentColors[l] || '#90A4AE'),
          borderWidth: 2, borderColor: '#fff' }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'left',
            labels: { padding: 16, usePointStyle: true, pointStyle: 'circle', font: { size: 13 } }
          }
        }
      }
    });

    // Top districts
    const distEl = document.getElementById('topDistricts');
    if (data.top_districts?.length) {
      distEl.innerHTML = '<table style="width:100%;border-collapse:collapse;"><thead><tr style="text-align:left;border-bottom:1px solid rgba(0,0,0,0.1);"><th style="padding:0.5rem;">District</th><th style="padding:0.5rem;text-align:right;">Grievances</th></tr></thead><tbody>' +
        data.top_districts.map(d => `<tr style="border-bottom:1px solid rgba(0,0,0,0.05);"><td style="padding:0.5rem;">${escapeHtml(d.district)}</td><td style="padding:0.5rem;text-align:right;font-weight:600;">${d.count}</td></tr>`).join('') +
        '</tbody></table>';
    } else { distEl.innerHTML = '<p class="text-muted">No data available.</p>'; }

    // Top complaints
    const compEl = document.getElementById('topComplaints');
    if (data.top_complaints?.length) {
      compEl.innerHTML = '<table style="width:100%;border-collapse:collapse;"><thead><tr style="text-align:left;border-bottom:1px solid rgba(0,0,0,0.1);"><th style="padding:0.5rem;">Complaint</th><th style="padding:0.5rem;">Dept</th><th style="padding:0.5rem;text-align:right;">Count</th></tr></thead><tbody>' +
        data.top_complaints.map(c => `<tr style="border-bottom:1px solid rgba(0,0,0,0.05);"><td style="padding:0.5rem;">${escapeHtml(c.title)}</td><td style="padding:0.5rem;">${deptBadge(c.department)}</td><td style="padding:0.5rem;text-align:right;font-weight:600;">${c.count}</td></tr>`).join('') +
        '</tbody></table>';
    } else { compEl.innerHTML = '<p class="text-muted">No data available.</p>'; }
  } catch (e) { showToast(e.message, 'error'); }
}
