<template>
  <footer class="navbar navbar-inverse">
    <ul>
      <li>
        <router-link to="">联系开发者</router-link>
      </li>
      <li>
        <router-link to="">服务条款</router-link>
      </li>
    </ul>
    <ul>
      <li>
        WSC-TOOL
      </li>
    </ul>
  </footer>
</template>

<script>
export default {
  name: 'Footer'
}
</script>

<style lang="scss" scoped>
@import "../assets/css/common.scss";
footer {
  height: $footerheight;
  margin-bottom: 0;
  position: absolute!important;
  bottom: 0;
  width: 100%;
  box-shadow: 0 0 9px 0 #0c0c0c;
  border-radius: 0px;

  ul {
    margin: 4px auto;
    text-align: center;
    li {
      list-style: none;
      display: inline;
      margin: 0 10px;
      a {
        text-decoration: none;
        color: #c0c0c0;
      }
    }
  }
}
</style>
