/**
 * Sabitler — Model isimlendirme haritası ve navigasyon tanımları
 */

export const MODEL_DISPLAY = {
  'baseline':                                       { short: 'Lojistik Regresyon',              badge: 'Temel',   type: 'Temel Model',      calibration: '—',         icon: '🔵' },
  'baseline_decision':                              { short: 'Lojistik Regresyon (Karar)',      badge: 'Temel',   type: 'Temel Model',      calibration: 'Karar Eşiği', icon: '🔵' },
  'baseline_calibrated_sigmoid':                    { short: 'Lojistik + Sigmoid Kalibrasyon',  badge: 'Temel',   type: 'Kalibre Model',    calibration: 'Sigmoid',   icon: '🟢' },
  'baseline_calibrated_sigmoid_decision':           { short: 'Lojistik + Sigmoid (Karar)',      badge: 'Temel',   type: 'Kalibre Model',    calibration: 'Sigmoid',   icon: '🟢' },
  'baseline_calibrated_isotonic':                   { short: 'Lojistik + İzotonik Kalibrasyon', badge: 'Temel',   type: 'Kalibre Model',    calibration: 'İzotonik',  icon: '🟢' },
  'baseline_calibrated_isotonic_decision':          { short: 'Lojistik + İzotonik (Karar)',     badge: 'Temel',   type: 'Kalibre Model',    calibration: 'İzotonik',  icon: '🟢' },
  'challenger_xgboost':                             { short: 'XGBoost',                         badge: 'Gelişmiş', type: 'Gelişmiş Model',  calibration: '—',         icon: '🟠' },
  'challenger_xgboost_decision':                    { short: 'XGBoost (Karar)',                 badge: 'Gelişmiş', type: 'Gelişmiş Model',  calibration: 'Karar Eşiği', icon: '🟠' },
  'challenger_xgboost_calibrated_sigmoid':          { short: 'XGBoost + Sigmoid Kalibrasyon',   badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'Sigmoid', icon: '🟤' },
  'challenger_xgboost_calibrated_sigmoid_decision': { short: 'XGBoost + Sigmoid (Karar)',       badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'Sigmoid', icon: '🟤' },
  'challenger_xgboost_calibrated_isotonic':         { short: 'XGBoost + İzotonik Kalibrasyon',  badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'İzotonik', icon: '🟤' },
  'challenger_xgboost_calibrated_isotonic_decision':{ short: 'XGBoost + İzotonik (Karar)',      badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'İzotonik', icon: '🟤' },
};

export const NAV_ITEMS = [
  { key: 'overview', path: '/',         label: 'Genel Bakış',         desc: 'Aktif model ve özet göstergeler' },
  { key: 'models',   path: '/models',   label: 'Model Karşılaştırma', desc: 'Tüm modellerin detaylı analizi' },
  { key: 'pipeline', path: '/pipeline', label: 'Veri İşleme Hattı',  desc: 'Önişleme, özellik çıkarımı ve model eğitim adımları' },
  { key: 'runs',     path: '/runs',     label: 'Koşu Geçmişi',       desc: 'Geçmiş çalıştırma kayıtları' },
  { key: 'guests', path: '/guests', label: 'Misafirler',        desc: 'Misafir kayıt, arama ve güncelleme' },
  { key: 'chat',   path: '/chat',   label: 'Misafir & Chat',    desc: 'Misafir kayıt, liste ve iptal azaltma asistanı' },

  { key: 'system', path: '/system', label: 'Sistem Durumu',     desc: 'Veritabanı ve altyapı bilgisi' },
];
