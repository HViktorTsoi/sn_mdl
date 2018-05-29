<template>
  <div id="app">
    <el-container>
      <el-header>
        <el-menu class="navbar" mode="horizontal">
          <el-menu-item class="navbar-brand" index="0">
            <i class="el-icon-share"></i>NME在线社交网络演化模型</el-menu-item>
        </el-menu>
      </el-header>
      <el-container>
        <el-main>
          <router-view></router-view>
        </el-main>
        <el-aside class="side-menu-container" width="200px">
          <el-menu text-color="#300232" active-text-color="#c43235" default-active="2">
            <el-menu-item index="2" @click="$router.push('/evolution')">
              <i class="el-icon-share"></i>
              <span slot="title">模型演化分析</span>
            </el-menu-item>
            <el-menu-item index="3" @click="$router.push('/comparison')">
              <i class="el-icon-news"></i>
              <span slot="title">数据集拟合</span>
            </el-menu-item>
            <el-menu-item index="4" @click="$router.push('setting')">
              <i class="el-icon-setting"></i>
              <span slot="title">用户设置</span>
            </el-menu-item>
          </el-menu>
        </el-aside>
      </el-container>
    </el-container>
  </div>
</template>

<script>

export default {
  name: 'app',
  mounted: function () {
    console.log(this.$route)
  }
}
</script>

<style lang="scss">
@import "./assets/css/common.scss";
body {
  margin: 0;
  padding: 0;
  .side-menu-container {
    .el-menu {
      border-left: solid 1px #e6e6e6;
      .el-menu-item {
        &:hover,
        &.is-active {
          background-color: $clr_w !important;
        }
      }
    }
  }
  .el-header {
    padding: 0;
  }
}
.navbar {
  background-color: $clr_b;
  .navbar-brand {
    &:hover {
      background-color: $clr_b !important;
    }
    &.is-active {
      background-color: $clr_b !important;
    }
    color: $clr_w !important;
    font-size: 16px;
    i {
      font-size: 23px !important;
      color: $clr_r !important;
    }
  }
}
</style>
